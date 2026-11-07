import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import {
  detectRelease, isPublished, parseReleaseSubject, syncVersion, verifyTarball,
} from './release.mjs';

test('release subjects map to stable and prerelease tags', () => {
  assert.deepEqual(parseReleaseSubject('chore: release 1.2.3'), {
    version: '1.2.3', tag: 'latest', prerelease: false,
  });
  assert.deepEqual(parseReleaseSubject('chore: release 1.2.3-rc.2'), {
    version: '1.2.3-rc.2', tag: 'rc', prerelease: true,
  });
  assert.deepEqual(parseReleaseSubject('chore: release 1.2.3-beta.0'), {
    version: '1.2.3-beta.0', tag: 'beta', prerelease: true,
  });
  assert.equal(parseReleaseSubject('fix: adjust tests'), null);
  for (const subject of [
    'chore: release 01.2.3',
    'chore: release 1.2.3-alpha',
    'chore: release 1.2.3-rc.01',
    'chore: release 1.2.3-latest.1',
    'chore: release 1.2.3 trailing',
  ]) assert.throws(() => parseReleaseSubject(subject));
});

test('only owner-originated pushes to main can trigger publishing', () => {
  const context = {
    eventName: 'push', branch: 'main', repository: 'twlite/recommendtf',
    actorId: '46562212', triggeringActor: 'twlite',
    subject: 'chore: release 0.2.0',
  };
  assert.equal(detectRelease(context)?.version, '0.2.0');
  for (const changes of [
    { eventName: 'pull_request' },
    { branch: 'feature' },
    { repository: 'fork/recommendtf' },
    { actorId: '123456' },
    { triggeringActor: 'other' },
  ]) assert.equal(detectRelease({ ...context, ...changes }), null);
});

test('sync modifies manifest version without affecting other fields', () => {
  const dir = mkdtempSync(join(tmpdir(), 'recommendtf-release-'));
  const path = join(dir, 'package.json');
  try {
    writeFileSync(path, JSON.stringify({ name: 'recommendtf', version: '0.1.3', scripts: { build: 'tsc' } }));
    const manifest = syncVersion('0.2.0-rc.1', path);
    assert.equal(manifest.version, '0.2.0-rc.1');
    assert.equal(JSON.parse(readFileSync(path, 'utf8')).scripts.build, 'tsc');
  } finally { rmSync(dir, { force: true, recursive: true }); }
});

test('npm existence probes distinguish missing and transient errors', async () => {
  assert.equal(await isPublished('recommendtf', '1.2.3', async () => ({ status: 404 })), false);
  assert.equal(await isPublished('recommendtf', '1.2.3', async () => ({
    status: 200, ok: true, json: async () => ({ name: 'recommendtf', version: '1.2.3' }),
  })), true);
  await assert.rejects(isPublished('recommendtf', '1.2.3',
    async () => ({ status: 503, ok: false })), /503/);
  await assert.rejects(isPublished('recommendtf', '1.2.3', async () => ({
    status: 200, ok: true, json: async () => ({ name: 'other', version: '1.2.3' }),
  })), /identity/);
});

test('archive verification rejects wrong version and missing entrypoints', () => {
  const dir = mkdtempSync(join(tmpdir(), 'recommendtf-pack-'));
  try {
    const packageRoot = join(dir, 'package');
    mkdirSync(join(packageRoot, 'dist'), { recursive: true });
    writeFileSync(join(packageRoot, 'package.json'), JSON.stringify({
      name: 'recommendtf', version: '0.2.0',
      main: 'dist/index.js', types: 'dist/index.d.ts',
    }));
    writeFileSync(join(packageRoot, 'dist/index.js'), 'module.exports = {};');
    writeFileSync(join(packageRoot, 'dist/index.d.ts'), 'export {};');
    const archive = join(dir, 'recommendtf-0.2.0.tgz');
    execFileSync('tar', ['-czf', archive, '-C', dir, 'package']);
    assert.equal(verifyTarball(archive, '0.2.0').name, 'recommendtf');
    assert.throws(() => verifyTarball(archive, '0.2.1'), /mismatch/);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
