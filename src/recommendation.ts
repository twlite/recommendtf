import type tensorflow from '@tensorflow/tfjs';
import { Id, StorageStrategy } from './storage';
import { InMemoryStorage } from './memory-storage';
import { loadTensorflow } from './load-tensorflow';

let tf: typeof tensorflow;

export type FeedbackMode = 'explicit' | 'implicit';

export interface Interaction {
  user: Id;
  entity: Id;
  /** Explicit score, or 0/1 in implicit mode. Defaults to 1. */
  rating?: number;
}

export interface RecommendationOptions {
  epoch?: number;
  embeddingSize?: number;
  learningRate?: number;
  batchSize?: number;
  regularization?: number;
  /** Use ratings (explicit) or binary interactions with sampled negatives (implicit). */
  feedback?: FeedbackMode;
  /** Number of unobserved items sampled per positive implicit interaction. */
  negativeSamples?: number;
  storage?: StorageStrategy;
  /** Kept for compatibility. TensorFlow.js backend selection is not changed by this option. */
  disableIsNode?: boolean;
}

export interface RecommendationQueryOptions {
  /** Exclude previously observed user/entity pairs. Defaults to true for getEntities. */
  excludeSeen?: boolean;
  /** Additional IDs to exclude. */
  exclude?: Iterable<Id>;
  /** Behavior when the requested user is unknown. Defaults to popular. */
  coldStart?: 'popular' | 'empty';
}

export interface SerializedModel {
  config: {
    epoch: number;
    embeddingSize: number;
    learningRate: number;
    batchSize: number;
    regularization?: number;
    feedback?: FeedbackMode;
    negativeSamples?: number;
  };
  userEmbeddings: number[][];
  entityEmbeddings: number[][];
  userMap: Array<[Id, number]>;
  entityMap: Array<[Id, number]>;
  reverseUserMap: Id[];
  reverseEntityMap: Id[];
  /** Added in v2; legacy exports are still accepted. */
  interactions?: Array<[Id, Id, number]>;
  initialized: boolean;
  version: string;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(name + ' must be a positive integer');
  }
  return value;
}

function validId(id: unknown): id is Id {
  return typeof id === 'string' || (typeof id === 'number' && Number.isFinite(id));
}

function validateMatrix(values: unknown, rows: number, columns: number, label: string): asserts values is number[][] {
  if (!Array.isArray(values) || values.length !== rows ||
    values.some((row) => !Array.isArray(row) || row.length !== columns ||
      row.some((value) => typeof value !== 'number' || !Number.isFinite(value)))) {
    throw new Error('Invalid ' + label + ' in serialized model');
  }
}

/**
 * Small-scale matrix-factorization recommender. The instance serializes reads and writes,
 * so training, resizing, export, and inference cannot race with one another.
 */
export class Recommendation {
  public storage: StorageStrategy;
  public userEmbeddings!: tensorflow.Variable;
  public entityEmbeddings!: tensorflow.Variable;
  public optimizer: tensorflow.Optimizer;

  private readonly embeddingSize: number;
  private readonly learningRate: number;
  private readonly epoch: number;
  private readonly batchSize: number;
  private readonly regularization: number;
  private readonly feedback: FeedbackMode;
  private readonly negativeSamples: number;
  private readonly history = new Map<Id, Map<Id, number>>();
  private operations: Promise<void> = Promise.resolve();
  private closed = false;
  private _initialized = false;

  constructor(options: RecommendationOptions = {}) {
    this.epoch = positiveInteger(options.epoch ?? 20, 'epoch');
    this.embeddingSize = positiveInteger(options.embeddingSize ?? 16, 'embeddingSize');
    this.batchSize = positiveInteger(options.batchSize ?? 256, 'batchSize');
    this.learningRate = options.learningRate ?? 0.01;
    this.regularization = options.regularization ?? 0.0001;
    this.negativeSamples = options.negativeSamples ?? 2;
    this.feedback = options.feedback ?? 'explicit';

    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0 ||
      !Number.isFinite(this.regularization) || this.regularization < 0 ||
      !Number.isSafeInteger(this.negativeSamples) || this.negativeSamples < 0 ||
      (this.feedback !== 'explicit' && this.feedback !== 'implicit')) {
      throw new RangeError('Invalid learningRate, regularization, negativeSamples, or feedback');
    }

    this.storage = options.storage ?? new InMemoryStorage();
    if (!tf) tf = loadTensorflow();
    // Do not mutate tf.ENV: it is process-global, and IS_NODE is not a warning toggle.
    this.optimizer = tf.train.adam(this.learningRate);
  }

  get initialized(): boolean {
    return this._initialized;
  }

  /** Queue all operations, including reads, to prevent disposal/resizing races. */
  private run<T>(operation: () => T | Promise<T>): Promise<T> {
    if (this.closed) return Promise.reject(new Error('Recommendation has been disposed'));
    const result = this.operations.then(operation);
    this.operations = result.then(() => undefined, () => undefined);
    return result;
  }

  private makeEmbeddings(rows: number): tensorflow.Variable {
    const initial = tf.randomNormal([rows, this.embeddingSize], 0, 0.25);
    try {
      return tf.variable(initial);
    } finally {
      initial.dispose();
    }
  }

  private async initializeOrResize(): Promise<void> {
    const users = (await this.storage.getAllUsers()).length;
    const entities = (await this.storage.getAllEntities()).length;

    if (!this._initialized) {
      this.userEmbeddings = this.makeEmbeddings(users);
      this.entityEmbeddings = this.makeEmbeddings(entities);
      this._initialized = true;
      return;
    }

    let resized = false;
    if (users > this.userEmbeddings.shape[0]) {
      const more = this.makeEmbeddings(users - this.userEmbeddings.shape[0]);
      const combined = tf.tidy(() => tf.concat([this.userEmbeddings, more], 0));
      more.dispose();
      const next = tf.variable(combined);
      combined.dispose();
      this.userEmbeddings.dispose();
      this.userEmbeddings = next;
      resized = true;
    }
    if (entities > this.entityEmbeddings.shape[0]) {
      const more = this.makeEmbeddings(entities - this.entityEmbeddings.shape[0]);
      const combined = tf.tidy(() => tf.concat([this.entityEmbeddings, more], 0));
      more.dispose();
      const next = tf.variable(combined);
      combined.dispose();
      this.entityEmbeddings.dispose();
      this.entityEmbeddings = next;
      resized = true;
    }

    if (resized) {
      // Adam's slots belong to the old variables, so they must be rebuilt.
      this.optimizer.dispose();
      this.optimizer = tf.train.adam(this.learningRate);
    }
  }

  async fit(data: Interaction[] = []): Promise<void> {
    return this.run(() => this.fitInternal(data));
  }

  private async fitInternal(data: Interaction[]): Promise<void> {
    if (!Array.isArray(data)) throw new TypeError('Interactions must be an array');
    if (data.length === 0) return;

    for (const interaction of data) {
      if (!interaction || !validId(interaction.user) || !validId(interaction.entity) ||
        !Number.isFinite(interaction.rating ?? 1)) {
        throw new TypeError('Each interaction needs valid user, entity, and finite rating');
      }
      if (this.feedback === 'implicit' && interaction.rating !== undefined &&
        interaction.rating !== 0 && interaction.rating !== 1) {
        throw new RangeError('Implicit feedback ratings must be 0 or 1');
      }
    }

    for (const { user, entity } of data) {
      await this.storage.getOrCreateUserIndex(user);
      await this.storage.getOrCreateEntityIndex(entity);
    }
    await this.initializeOrResize();

    // Record observations before negative sampling so positive events in this call
    // cannot accidentally be selected as negative examples.
    for (const { user, entity, rating } of data) {
      let observed = this.history.get(user);
      if (!observed) {
        observed = new Map();
        this.history.set(user, observed);
      }
      observed.set(entity, rating ?? 1);
    }

    const allEntities = this.feedback === 'implicit' ? await this.storage.getAllEntities() : [];
    const order = data.map((_, index) => index);

    for (let epoch = 0; epoch < this.epoch; epoch++) {
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }

      for (let start = 0; start < order.length; start += this.batchSize) {
        const userIndices: number[] = [];
        const entityIndices: number[] = [];
        const targets: number[] = [];
        const end = Math.min(start + this.batchSize, order.length);

        for (let i = start; i < end; i++) {
          const interaction = data[order[i]];
          const userIndex = await this.storage.getUserIndex(interaction.user);
          const entityIndex = await this.storage.getEntityIndex(interaction.entity);
          if (userIndex == null || entityIndex == null) throw new Error('Storage mapping changed during training');

          const rating = interaction.rating ?? 1;
          userIndices.push(userIndex);
          entityIndices.push(entityIndex);
          targets.push(rating);

          if (this.feedback === 'implicit' && rating === 1 && this.negativeSamples > 0) {
            const observed = this.history.get(interaction.user)!;
            if (observed.size >= allEntities.length) continue;
            for (let n = 0; n < this.negativeSamples; n++) {
              let candidate = Math.floor(Math.random() * allEntities.length);
              let tries = 0;
              while (observed.has(allEntities[candidate]) && tries++ < 16) {
                candidate = Math.floor(Math.random() * allEntities.length);
              }
              if (observed.has(allEntities[candidate])) {
                const first = candidate;
                do {
                  candidate = (candidate + 1) % allEntities.length;
                } while (candidate !== first && observed.has(allEntities[candidate]));
                if (observed.has(allEntities[candidate])) break;
              }
              userIndices.push(userIndex);
              entityIndices.push(candidate);
              targets.push(0);
            }
          }
        }

        const userTensor = tf.tensor1d(userIndices, 'int32');
        const entityTensor = tf.tensor1d(entityIndices, 'int32');
        const targetsTensor = tf.tensor1d(targets);
        try {
          this.optimizer.minimize(() => tf.tidy(() => {
            const users = tf.gather(this.userEmbeddings, userTensor);
            const entities = tf.gather(this.entityEmbeddings, entityTensor);
            const predictions = tf.sum(tf.mul(users, entities), 1);
            const error = this.feedback === 'explicit'
              ? tf.mean(tf.square(tf.sub(predictions, targetsTensor)))
              : tf.mean(tf.softplus(tf.mul(predictions, tf.sub(1, tf.mul(targetsTensor, 2)))));
            if (this.regularization === 0) return error;
            const penalty = tf.mul(this.regularization, tf.add(
              tf.mean(tf.square(users)),
              tf.mean(tf.square(entities))
            ));
            return tf.add(error, penalty) as tensorflow.Scalar;
          }), false, [this.userEmbeddings, this.entityEmbeddings]);
        } finally {
          userTensor.dispose();
          entityTensor.dispose();
          targetsTensor.dispose();
        }

        // Yield periodically; fit() is async but the TensorFlow CPU backend is synchronous.
        if (start > 0 && start % (this.batchSize * 10) === 0) {
          await new Promise<void>((resolve) => setTimeout(resolve, 0));
        }
      }
    }
  }

  private async scoreAll(matrix: tensorflow.Variable, index: number, other: tensorflow.Variable): Promise<number[]> {
    const scores = tf.tidy(() => {
      const selected = tf.gather(matrix, tf.tensor1d([index], 'int32'));
      return tf.matMul(other, selected, false, true).reshape([-1]);
    });
    try {
      const values = Array.from(await scores.data());
      return this.feedback === 'implicit'
        ? values.map((score) => 1 / (1 + Math.exp(-score)))
        : values;
    } finally {
      scores.dispose();
    }
  }

  private top(scores: number[], ids: Id[], count: number, ignored: Set<Id>): Id[] {
    return ids
      .map((id, index) => ({ id, score: scores[index] }))
      .filter((entry) => !ignored.has(entry.id) && Number.isFinite(entry.score))
      .sort((a, b) => b.score - a.score)
      .slice(0, count)
      .map((entry) => entry.id);
  }

  private popular(count: number, ignored: Set<Id>): Id[] {
    const scores = new Map<Id, { count: number; total: number }>();
    for (const seen of this.history.values()) {
      for (const [entity, rating] of seen) {
        const current = scores.get(entity) ?? { count: 0, total: 0 };
        current.count++;
        current.total += rating;
        scores.set(entity, current);
      }
    }
    return Array.from(scores)
      .filter(([id]) => !ignored.has(id))
      .sort((a, b) => b[1].count - a[1].count ||
        b[1].total / b[1].count - a[1].total / a[1].count)
      .slice(0, count)
      .map(([id]) => id);
  }

  async getEntities(user: Id, count = 10, options: RecommendationQueryOptions = {}): Promise<Id[]> {
    return this.run(async () => {
      if (!Number.isSafeInteger(count) || count < 0) throw new RangeError('count must be nonnegative');
      if (count === 0 || !this._initialized) return [];
      const ignored = new Set(options.exclude ?? []);
      if (options.excludeSeen !== false) {
        for (const id of this.history.get(user)?.keys() ?? []) ignored.add(id);
      }
      const index = await this.storage.getUserIndex(user);
      if (index == null) return options.coldStart === 'empty' ? [] : this.popular(count, ignored);
      return this.top(await this.scoreAll(this.userEmbeddings, index, this.entityEmbeddings),
        await this.storage.getAllEntities(), count, ignored);
    });
  }

  async getUsers(entity: Id, count = 10, options: RecommendationQueryOptions = {}): Promise<Id[]> {
    return this.run(async () => {
      if (!Number.isSafeInteger(count) || count < 0) throw new RangeError('count must be nonnegative');
      if (count === 0 || !this._initialized) return [];
      const index = await this.storage.getEntityIndex(entity);
      if (index == null) return [];
      const ignored = new Set(options.exclude ?? []);
      if (options.excludeSeen) {
        for (const [user, seen] of this.history) {
          if (seen.has(entity)) ignored.add(user);
        }
      }
      return this.top(await this.scoreAll(this.entityEmbeddings, index, this.userEmbeddings),
        await this.storage.getAllUsers(), count, ignored);
    });
  }

  /** Predict an explicit rating or an implicit interaction probability. */
  async predict(user: Id, entity: Id): Promise<number | null> {
    return this.run(async () => {
      if (!this._initialized) return null;
      const userIndex = await this.storage.getUserIndex(user);
      const entityIndex = await this.storage.getEntityIndex(entity);
      if (userIndex == null || entityIndex == null) return null;
      const scores = await this.scoreAll(this.userEmbeddings, userIndex, this.entityEmbeddings);
      return scores[entityIndex] ?? null;
    });
  }

  private async exportInternal(): Promise<SerializedModel> {
    if (!this._initialized) throw new Error('Call fit() before exporting');
    const mapping = await this.storage.exportData();
    const interactions: Array<[Id, Id, number]> = [];
    for (const [user, entities] of this.history) {
      for (const [entity, rating] of entities) interactions.push([user, entity, rating]);
    }
    return {
      config: {
        epoch: this.epoch, embeddingSize: this.embeddingSize,
        learningRate: this.learningRate, batchSize: this.batchSize,
        regularization: this.regularization, feedback: this.feedback,
        negativeSamples: this.negativeSamples,
      },
      userEmbeddings: await this.userEmbeddings.array() as number[][],
      entityEmbeddings: await this.entityEmbeddings.array() as number[][],
      ...mapping,
      interactions,
      initialized: true,
      version: '2.0.0',
    };
  }

  async export(): Promise<SerializedModel> {
    return this.run(() => this.exportInternal());
  }

  /** Write a complete snapshot to a temporary file, then atomically rename it. */
  async save(filePath: string): Promise<void> {
    return this.run(async () => {
      const fs = await import('node:fs/promises');
      const path = await import('node:path');
      const crypto = await import('node:crypto');
      const snapshot = await this.exportInternal();
      await fs.mkdir(path.dirname(filePath), { recursive: true });
      const temporary = filePath + '.' + crypto.randomUUID() + '.tmp';
      try {
        await fs.writeFile(temporary, JSON.stringify(snapshot), { encoding: 'utf8', flag: 'wx' });
        await fs.rename(temporary, filePath);
      } finally {
        await fs.rm(temporary, { force: true });
      }
    });
  }

  static async from(model: SerializedModel, storage?: StorageStrategy): Promise<Recommendation> {
    if (!model || !model.config || !Array.isArray(model.reverseUserMap) ||
      !Array.isArray(model.reverseEntityMap) || !Array.isArray(model.userMap) ||
      !Array.isArray(model.entityMap) || model.initialized !== true ||
      (model.version !== '1.0.0' && model.version !== '2.0.0')) {
      throw new Error('Invalid or unsupported serialized model');
    }
    const recommendation = new Recommendation({ ...model.config, storage });
    try {
      validateMatrix(model.userEmbeddings, model.reverseUserMap.length,
        recommendation.embeddingSize, 'user embeddings');
      validateMatrix(model.entityEmbeddings, model.reverseEntityMap.length,
        recommendation.embeddingSize, 'entity embeddings');

      if (model.userMap.length !== model.reverseUserMap.length ||
        model.entityMap.length !== model.reverseEntityMap.length ||
        model.userMap.some(([id, index]) => !validId(id) ||
          !Number.isSafeInteger(index) || model.reverseUserMap[index] !== id) ||
        model.entityMap.some(([id, index]) => !validId(id) ||
          !Number.isSafeInteger(index) || model.reverseEntityMap[index] !== id)) {
        throw new Error('Invalid ID mappings in serialized model');
      }

      await recommendation.storage.importData({
        userMap: model.userMap, entityMap: model.entityMap,
        reverseUserMap: model.reverseUserMap, reverseEntityMap: model.reverseEntityMap,
      });

      const users = tf.tensor2d(model.userEmbeddings,
        [model.reverseUserMap.length, recommendation.embeddingSize]);
      const entities = tf.tensor2d(model.entityEmbeddings,
        [model.reverseEntityMap.length, recommendation.embeddingSize]);
      try {
        recommendation.userEmbeddings = tf.variable(users);
        recommendation.entityEmbeddings = tf.variable(entities);
      } finally {
        users.dispose();
        entities.dispose();
      }
      recommendation._initialized = true;

      if (model.interactions !== undefined) {
        if (!Array.isArray(model.interactions)) throw new Error('Invalid interaction history');
        for (const entry of model.interactions) {
          if (!Array.isArray(entry) || entry.length !== 3 ||
            !validId(entry[0]) || !validId(entry[1]) ||
            typeof entry[2] !== 'number' || !Number.isFinite(entry[2])) {
            throw new Error('Invalid interaction history entry');
          }
          let seen = recommendation.history.get(entry[0]);
          if (!seen) {
            seen = new Map();
            recommendation.history.set(entry[0], seen);
          }
          seen.set(entry[1], entry[2]);
        }
      }
      return recommendation;
    } catch (error) {
      await recommendation.dispose();
      throw error;
    }
  }

  static async fromPath(filePath: string, storage?: StorageStrategy): Promise<Recommendation> {
    const fs = await import('node:fs/promises');
    const text = await fs.readFile(filePath, 'utf8');
    return Recommendation.from(JSON.parse(text) as SerializedModel, storage);
  }

  /** Release variables, optimizer slots, and any future tensor state. */
  async dispose(): Promise<void> {
    if (this.closed) return this.operations;
    this.closed = true;
    const result = this.operations.then(() => {
      if (this._initialized) {
        this.userEmbeddings?.dispose();
        this.entityEmbeddings?.dispose();
      }
      this.optimizer.dispose();
      this._initialized = false;
      this.history.clear();
    });
    this.operations = result.then(() => undefined, () => undefined);
    return result;
  }
}
