import { execFileSync, spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('../../', import.meta.url)));
export const PACKAGE_NAME = 'recommendtf';
export const OFFICIAL_REPOSITORY = 'twlite/recommendtf';
export const RELEASE_ACTOR_ID = '46562212';
const SUBJECT_PREFIX = 'chore: release';
const SUBJECT_PATTERN =
  /^chore: release ((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-([a-z][a-z0-9-]*)\.(?:0|[1-9]\d*))?)$/;

export function parseReleaseSubject(subject) {
  if (typeof subject !== 'string') throw new TypeError('Commit subject must be a string');
  if (!subject.startsWith(SUBJECT_PREFIX)) return null;
  const match = SUBJECT_PATTERN.exec(subject);
  if (!match || match[2] === 'latest') {
    throw new Error(
      'Invalid release subject. Use "chore: release X.Y.Z" or "chore: release X.Y.Z-(dev|beta|rc).N" (other lowercase channels are also supported).'
    );
  }
  const version = match[1];
  const tag = match[2] ?? 'latest';
  return { version, tag, prerelease: tag !== 'latest' };
}

export function detectRelease({ eventName, branch, repository, actorId, triggeringActor, subject }) {
  if ((eventName !== 'push' && eventName !== 'workflow_dispatch') ||
    branch !== 'main' || repository !== OFFICIAL_REPOSITORY ||
    String(actorId) !== RELEASE_ACTOR_ID || triggeringActor !== 'twlite') {
    return null;
  }
  return parseReleaseSubject(subject);
}

function requireReleaseVersion(version) {
  const result = parseReleaseSubject('chore: release ' + version);
  if (!result) throw new Error('Invalid release version: ' + version);
  return result;
}

function subjectFromHead() {
  return execFileSync('git', ['log', '-1', '--format=%s'], {
    cwd: root, encoding: 'utf8',
  }).trim();
}

export function syncVersion(version, packageFile = join(root, 'package.json')) {
  requireReleaseVersion(version);
  const manifest = JSON.parse(readFileSync(packageFile, 'utf8'));
  if (manifest.name !== PACKAGE_NAME) throw new Error('Unexpected package name');
  manifest.version = version;
  writeFileSync(packageFile, JSON.stringify(manifest, null, 2) + '\n');
  return manifest;
}

export function verifyTarball(archive, version) {
  if (!existsSync(archive)) throw new Error('Release tarball missing: ' + archive);
  const content = execFileSync('tar', ['-xOzf', archive, 'package/package.json'], {
    encoding: 'utf8',
  });
  const manifest = JSON.parse(content);
  if (manifest.name !== PACKAGE_NAME || manifest.version !== version ||
    !manifest.main || !manifest.types) {
    throw new Error('Release tarball metadata mismatch');
  }
  // Verify compiled entrypoints are included in the exact archive we publish.
  const entries = execFileSync('tar', ['-tzf', archive], { encoding: 'utf8' })
    .trim().split('\n');
  for (const name of [manifest.main, manifest.types]) {
    if (!entries.includes('package/' + name.replace(/^\.\//, ''))) {
      throw new Error('Release tarball lacks ' + name);
    }
  }
  return manifest;
}

export async function isPublished(name, version, fetchImpl = fetch) {
  const response = await fetchImpl(
    'https://registry.npmjs.org/' + encodeURIComponent(name) + '/' + encodeURIComponent(version),
    { headers: { accept: 'application/json' }, signal: AbortSignal.timeout(20_000) }
  );
  if (response.status === 404) return false;
  if (!response.ok) {
    throw new Error('npm registry lookup failed: HTTP ' + response.status);
  }
  const published = await response.json();
  if (published.name !== name || published.version !== version) {
    throw new Error('Unexpected package identity from npm registry');
  }
  return true;
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { cwd: root, encoding: 'utf8', stdio: 'inherit', ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(command + ' failed with exit status ' + result.status);
}

async function publish(version, tag, directory) {
  const archive = join(resolve(directory), PACKAGE_NAME + '-' + version + '.tgz');
  verifyTarball(archive, version);
  if (await isPublished(PACKAGE_NAME, version)) {
    console.log(PACKAGE_NAME + '@' + version + ' already exists on npm; skipping.');
    return;
  }
  // npm >=11.5.1 obtains short-lived credentials from GitHub's OIDC identity.
  // No NPM_TOKEN, NODE_AUTH_TOKEN, or npmrc authentication is used.
  run('npm', ['publish', archive, '--access', 'public', '--tag', tag, '--provenance']);
}

function releaseOnGitHub(version, prerelease) {
  const tag = 'v' + version;
  const ref = 'refs/tags/' + tag;
  const remote = spawnSync('git', ['ls-remote', '--exit-code', '--refs', 'origin', ref], {
    cwd: root, encoding: 'utf8',
  });
  if (remote.error) throw remote.error;
  if (remote.status === 0) {
    run('git', ['fetch', '--no-tags', 'origin', ref + ':' + ref]);
    const commit = execFileSync('git', ['rev-parse', ref + '^{commit}'], {
      cwd: root, encoding: 'utf8',
    }).trim();
    if (commit !== process.env.GITHUB_SHA) {
      throw new Error('Refusing to reuse ' + tag + ': tag points to ' + commit);
    }
  } else if (remote.status !== 2) {
    throw new Error('Unable to check remote release tag ' + tag);
  }

  const existing = spawnSync('gh', ['release', 'view', tag, '--json', 'isDraft'], {
    cwd: root, encoding: 'utf8', env: process.env,
  });
  if (existing.error) throw existing.error;
  if (existing.status === 0) {
    const release = JSON.parse(existing.stdout);
    if (release.isDraft) {
      run('gh', ['release', 'edit', tag, '--draft=false', '--prerelease=' + String(prerelease)]);
    } else {
      console.log('GitHub release ' + tag + ' already exists; skipping.');
    }
    return;
  }
  if (!/not found|release not found|HTTP 404/i.test(existing.stderr)) {
    throw new Error('Could not query GitHub release: ' + existing.stderr.trim());
  }

  run('gh', ['release', 'create', tag, '--target', process.env.GITHUB_SHA,
    '--title', tag, '--generate-notes', ...(prerelease ? ['--prerelease'] : ['--latest'])]);
}

async function main() {
  const [command, ...args] = process.argv.slice(2);
  if (command === 'detect') {
    const release = detectRelease({
      eventName: process.env.GITHUB_EVENT_NAME,
      branch: process.env.GITHUB_REF_NAME,
      repository: process.env.GITHUB_REPOSITORY,
      actorId: process.env.GITHUB_ACTOR_ID,
      triggeringActor: process.env.GITHUB_TRIGGERING_ACTOR,
      subject: subjectFromHead(),
    });
    const output = [
      'is_release=' + Boolean(release),
      'version=' + (release?.version ?? ''),
      'npm_tag=' + (release?.tag ?? ''),
      'prerelease=' + (release?.prerelease ?? false),
    ].join('\n') + '\n';
    console.log(output);
    if (process.env.GITHUB_OUTPUT) {
      const { appendFileSync } = await import('node:fs');
      appendFileSync(process.env.GITHUB_OUTPUT, output);
    }
    return;
  }

  if (command === 'sync') {
    syncVersion(args[0] ?? process.env.RELEASE_VERSION);
    return;
  }

  if (command === 'verify') {
    verifyTarball(args[0], process.env.RELEASE_VERSION);
    return;
  }

  if (command === 'publish') {
    const { version, tag } = requireReleaseVersion(process.env.RELEASE_VERSION);
    if (tag !== process.env.NPM_TAG) throw new Error('Release tag mismatch');
    await publish(version, tag, args[0]);
    return;
  }

  if (command === 'github-release') {
    const { version, prerelease } = requireReleaseVersion(process.env.RELEASE_VERSION);
    if (process.env.GITHUB_REPOSITORY !== OFFICIAL_REPOSITORY ||
      process.env.GITHUB_ACTOR_ID !== RELEASE_ACTOR_ID ||
      process.env.GITHUB_TRIGGERING_ACTOR !== 'twlite' ||
      process.env.GITHUB_REF_NAME !== 'main') {
      throw new Error('Unauthorized release context');
    }
    releaseOnGitHub(version, prerelease);
    return;
  }

  throw new Error('Usage: release.mjs <detect|sync|verify|publish|github-release>');
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
