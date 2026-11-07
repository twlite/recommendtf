# recommendtf

Small-scale collaborative filtering with TensorFlow.js and TypeScript.

The model learns user and item embeddings using **vectorized mini-batch training**. It handles explicit ratings or implicit binary feedback, seen-item exclusion, incremental additions, persistence, and disposal.

## Install

```sh
npm install recommendtf @tensorflow/tfjs
```

On supported platforms you can install `@tensorflow/tfjs-node` for native TensorFlow performance. The library tries `@tensorflow/tfjs-node-gpu`, `@tensorflow/tfjs-node`, then `@tensorflow/tfjs`.

## Explicit ratings

```ts
import { Recommendation } from 'recommendtf';

const model = new Recommendation({
  epoch: 20,
  embeddingSize: 16,
  batchSize: 256,
  learningRate: 0.01,
  regularization: 0.0001,
});

await model.fit([
  { user: 'alice', entity: 'film-a', rating: 5 },
  { user: 'alice', entity: 'film-b', rating: 1 },
  { user: 'bob', entity: 'film-a', rating: 4 },
  { user: 'bob', entity: 'film-c', rating: 5 },
]);

console.log(await model.getEntities('alice', 10));
console.log(await model.predict('alice', 'film-c'));
await model.save('./models/ratings.json');
await model.dispose();
```

By default `getEntities()` excludes items that the user has interacted with. Use `getEntities('alice', 10, { excludeSeen: false })` to include them. Use `getUsers(item, 10, { excludeSeen: true })` to exclude users who have interacted with an item.

New users receive popular items based on observed interactions, or nothing with `{ coldStart: 'empty' }`. A completely new item has no trained embedding and is not ranked until you add it using `fit()`.

## Implicit feedback

```ts
const model = new Recommendation({
  feedback: 'implicit',
  negativeSamples: 2,
  epoch: 15,
});

await model.fit([
  { user: 'alice', entity: 'post-1' },
  { user: 'alice', entity: 'post-2', rating: 0 },
  { user: 'bob', entity: 'post-3' },
]);

const recommendations = await model.getEntities('alice');
await model.dispose();
```

Implicit ratings must be 0 or 1, where 1 (also the default when omitted) is positive. Unobserved items are sampled as negative training examples. `predict()` returns a sigmoid-transformed score between 0 and 1; **it is not a calibrated probability**. Explicit mode predicts an unbounded rating.

## Persistence

```ts
const model = await Recommendation.fromPath('./models/ratings.json');
await model.fit([{ user: 'alice', entity: 'film-d', rating: 4 }]);
const snapshot = await model.export();
await model.save('./models/ratings.json');
await model.dispose();

const reloaded = await Recommendation.from(snapshot);
await reloaded.dispose();
```

JSON snapshots use format version 2.0.0 and include recorded interaction history. The 1.0.0 serialization format still imports, but it lacks interaction history, so filtering and popularity fallbacks only cover interactions added afterward. The numeric package version is independent of the snapshot format.

Resizing embeddings retains learned values but resets Adam's optimizer state. Saved snapshots also do not persist optimizer state. `save()` writes to a temporary file and renames it on completion.

## Production notes

- This is an in-process model for small catalogs, not an approximate nearest-neighbor service. Queries score and sort all known users or items.
- Training and inference are queued per instance, avoiding concurrent tensor disposal or embedding resize. Training will still block queries while it executes. Prefer periodic training and read-mostly snapshots for latency-sensitive APIs.
- New `fit()` calls train on supplied interactions only. For stable quality, periodically retrain on representative historical events held in your database.
- Track Recall@K, NDCG@K or rating RMSE on a held-out dataset and compare with a popularity baseline before deploying. The included regression tests do **not** establish recommendation quality.
- Use `dispose()` when replacing models. Model snapshots contain user identifiers and interaction history. Handle those files as user data.
- Custom `StorageStrategy` implementations must keep user/item indices contiguous and stable.
- The library cannot recommend new items without trained interactions or content features.

## API

`new Recommendation(options?)` accepts `epoch`, `embeddingSize`, `learningRate`, `batchSize`, `regularization`, `feedback`, `negativeSamples` and `storage`. The legacy `disableIsNode` option is accepted but no longer modifies the process-global TensorFlow environment.

Methods: `fit(interactions)`, `getEntities(user, count?, options?)`, `getUsers(item, count?, options?)`, `predict(user, item)`, `export()`, `save(path)`, `Recommendation.from(snapshot, storage?)`, `Recommendation.fromPath(path, storage?)`, `dispose()`.

Run `pnpm install --frozen-lockfile && pnpm test` to compile and execute the Node.js test suite.

MIT licensed.

## Maintainer release workflow

This repository uses pnpm for dependency installation, builds and tests. CI caches pnpm's content-addressable store, while `pnpm install --frozen-lockfile` rejects unsynchronized dependencies.

To release, push a commit to `main` whose **subject** is one of:

```
chore: release 0.2.0
chore: release 0.2.0-rc.1
chore: release 0.2.0-beta.1
```

The workflow runs tests on Node 20, 22 and 24, applies the release version without editing or committing the lockfile, builds and packs a verified tarball, then publishes through **npm trusted publishing over GitHub OIDC**. Already published exact versions are skipped, so rerunning a partial release is safe. A single GitHub release is created after npm publishing succeeds. Stable releases use the `latest` npm tag, while prereleases use their channel (for example `rc`).

**One-time npm setup:** An npm package owner must configure `recommendtf` under npm's **Trusted Publishers** settings with GitHub organization/user `twlite`, repository `recommendtf`, and workflow file `ci.yml` (the optional environment field is empty). This is required for publishing; the GitHub workflow intentionally does not use `NPM_TOKEN` or an npm auth token. The publish job grants `id-token: write` only where needed.

Only release commits originating from the repository owner account on `main` initiate publishing. Reruns must also be started by the owner. Do not use `chore: release ...` in unrelated commit subjects.
