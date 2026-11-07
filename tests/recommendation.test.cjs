const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const tf = require('@tensorflow/tfjs');
const { Recommendation } = require('../dist');

const data = [
  { user: 'a', entity: 'one', rating: 5 },
  { user: 'a', entity: 'two', rating: 4 },
  { user: 'b', entity: 'two', rating: 5 },
  { user: 'b', entity: 'three', rating: 4 },
  { user: 'c', entity: 'one', rating: 4 },
  { user: 'c', entity: 'three', rating: 5 },
];

test('explicit ratings, seen filtering and cold start', async () => {
  const rec = new Recommendation({ epoch: 8, embeddingSize: 4, batchSize: 2 });
  try {
    await rec.fit(data);
    assert(Number.isFinite(await rec.predict('a', 'one')));
    assert.equal(await rec.predict('missing', 'one'), null);
    assert.deepEqual(await rec.getEntities('a', 10), ['three']);
    assert.equal((await rec.getEntities('a', 10, { excludeSeen: false })).length, 3);
    assert.equal((await rec.getEntities('missing', 5)).length, 3);
    assert.deepEqual(await rec.getEntities('missing', 5, { coldStart: 'empty' }), []);
    assert.deepEqual(await rec.getUsers('unknown'), []);
    assert.equal((await rec.getUsers('one', 10, { excludeSeen: true })).includes('a'), false);
  } finally { await rec.dispose(); }
});

test('incremental growth, persistence and legacy import', async () => {
  const rec = new Recommendation({ epoch: 3, embeddingSize: 3, batchSize: 2 });
  const folder = await fs.mkdtemp(path.join(os.tmpdir(), 'recommendtf-'));
  try {
    await rec.fit(data);
    await rec.fit([{ user: 'new', entity: 'four', rating: 5 }]);
    assert.equal((await rec.getEntities('new', 10)).includes('four'), false);
    const before = await rec.predict('new', 'four');
    await rec.save(path.join(folder, 'model.json'));
    const restored = await Recommendation.fromPath(path.join(folder, 'model.json'));
    try {
      assert.equal(await restored.predict('new', 'four'), before);
      assert.equal((await restored.getEntities('new', 10)).includes('four'), false);
    } finally { await restored.dispose(); }
    const legacy = await rec.export();
    legacy.version = '1.0.0';
    delete legacy.interactions;
    delete legacy.config.feedback;
    delete legacy.config.negativeSamples;
    delete legacy.config.regularization;
    const old = await Recommendation.from(legacy);
    try { assert.equal(await old.predict('new', 'four'), before); }
    finally { await old.dispose(); }
  } finally {
    await rec.dispose();
    await fs.rm(folder, { recursive: true, force: true });
  }
});

test('implicit feedback with negative sampling', async () => {
  const rec = new Recommendation({ feedback: 'implicit', epoch: 8, batchSize: 2, embeddingSize: 4 });
  try {
    await rec.fit([
      { user: 'a', entity: 'one' }, { user: 'a', entity: 'two' },
      { user: 'b', entity: 'two' }, { user: 'b', entity: 'three' },
      { user: 'c', entity: 'four', rating: 0 },
    ]);
    const probability = await rec.predict('a', 'one');
    assert(probability >= 0 && probability <= 1);
    assert(!(await rec.getEntities('a', 10)).includes('one'));
  } finally { await rec.dispose(); }
});

test('invalid data and counts', async () => {
  const rec = new Recommendation({ epoch: 1 });
  try {
    await assert.rejects(rec.fit([{ user: 'a', entity: 'b', rating: Infinity }]));
    await assert.rejects(rec.getEntities('a', -1));
    await assert.rejects(rec.export());
  } finally { await rec.dispose(); }
  await assert.rejects(rec.fit(data), /disposed/);
});

test('repeated inference does not leak tensors', async () => {
  const rec = new Recommendation({ epoch: 3, embeddingSize: 4, batchSize: 3 });
  try {
    await rec.fit(data);
    const before = tf.memory().numTensors;
    for (let i = 0; i < 20; i++) {
      await rec.getEntities('a');
      await rec.getUsers('one');
    }
    assert.equal(tf.memory().numTensors, before);
  } finally { await rec.dispose(); }
});
