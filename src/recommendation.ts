import type tensorflow from '@tensorflow/tfjs';
import { Id, StorageStrategy } from './storage';
import { InMemoryStorage } from './memory-storage';
import { loadTensorflow } from './load-tensorflow';

let tf: typeof tensorflow;

/**
 * Serializable model data that can be saved to JSON.
 */
export interface SerializedModel {
  /** Model configuration options */
  config: {
    epoch: number;
    embeddingSize: number;
    learningRate: number;
    batchSize: number;
  };
  /** User embeddings as a 2D array */
  userEmbeddings: number[][];
  /** Entity embeddings as a 2D array */
  entityEmbeddings: number[][];
  /** User ID to index mapping */
  userMap: Array<[Id, number]>;
  /** Entity ID to index mapping */
  entityMap: Array<[Id, number]>;
  /** Reverse user mapping (index to ID) */
  reverseUserMap: Id[];
  /** Reverse entity mapping (index to ID) */
  reverseEntityMap: Id[];
  /** Whether the model has been initialized */
  initialized: boolean;
  /** Version for compatibility checking */
  version: string;
}

/**
 * Represents an interaction between a user and an entity.
 */
export interface Interaction {
  /** The unique identifier of the user */
  user: Id;
  /** The unique identifier of the entity */
  entity: Id;
  /** Optional rating for the interaction (defaults to 1 if not provided) */
  rating?: number;
}

/**
 * Configuration options for the recommendation system.
 */
export interface RecommendationOptions {
  /** Number of training epochs (default: 5) */
  epoch?: number;
  /** Size of the embedding vectors (default: 16) */
  embeddingSize?: number;
  /** Learning rate for the optimizer (default: 0.05) */
  learningRate?: number;
  /** Storage strategy for managing user and entity mappings (default: InMemoryStorage) */
  storage?: StorageStrategy;
  /** Batch size for processing data in chunks (default: 1000) */
  batchSize?: number;
  /** Whether to disable IS_NODE env (default: false) */
  disableIsNode?: boolean;
}

/**
 * A collaborative filtering recommendation system using TensorFlow.js.
 * Uses matrix factorization with embeddings to learn user and entity representations.
 */
export class Recommendation {
  public storage: StorageStrategy;
  public userEmbeddings!: tensorflow.Variable;
  public entityEmbeddings!: tensorflow.Variable;
  public optimizer: tensorflow.Optimizer;

  private embeddingSize: number;
  private learningRate: number;
  private epoch: number;
  private batchSize: number;
  private _initialized = false;

  /**
   * Creates a new recommendation system instance.
   * @param options - Configuration options for the recommendation system
   */
  public constructor(options: RecommendationOptions = {}) {
    this.epoch = options.epoch ?? 5;
    this.embeddingSize = options.embeddingSize ?? 16;
    this.learningRate = options.learningRate ?? 0.05;
    this.batchSize = options.batchSize ?? 1000;
    this.storage = options.storage ?? new InMemoryStorage();

    if (!tf) tf = loadTensorflow();

    if ('disableIsNode' in options) {
      tf.ENV.set('IS_NODE', !options.disableIsNode);
    }

    this.optimizer = tf.train.adam(this.learningRate);
  }

  public get initialized(): boolean {
    return this._initialized;
  }

  /**
   * Initializes the user and entity embedding matrices with random values in batches.
   * Called automatically when fitting data for the first time.
   * @private
   */
  private async initializeEmbeddings() {
    const numUsers = (await this.storage.getAllUsers()).length;
    const numEntities = (await this.storage.getAllEntities()).length;

    // Initialize embeddings in chunks to reduce memory usage
    this.userEmbeddings = await this.createEmbeddingsInBatches(
      numUsers,
      this.embeddingSize
    );
    this.entityEmbeddings = await this.createEmbeddingsInBatches(
      numEntities,
      this.embeddingSize
    );
    this._initialized = true;
  }

  /**
   * Creates embedding matrices in batches to reduce memory consumption.
   * @param totalSize - Total number of embeddings to create
   * @param embeddingDim - Dimension of each embedding vector
   * @returns Promise resolving to the created embedding variable
   * @private
   */
  private async createEmbeddingsInBatches(
    totalSize: number,
    embeddingDim: number
  ): Promise<tensorflow.Variable> {
    if (totalSize === 0) {
      return tf.variable(tf.zeros([0, embeddingDim]));
    }

    const batches: tensorflow.Tensor[] = [];
    const batchSize = Math.min(this.batchSize, totalSize);

    for (let i = 0; i < totalSize; i += batchSize) {
      const currentBatchSize = Math.min(batchSize, totalSize - i);
      const batch = tf.randomNormal([currentBatchSize, embeddingDim]);
      batches.push(batch);

      // Add a small delay to prevent blocking the event loop
      if (batches.length % 10 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 1));
      }
    }

    const fullMatrix = tf.concat(batches, 0);

    // Clean up intermediate tensors
    batches.forEach((batch) => batch.dispose());

    return tf.variable(fullMatrix);
  }

  /**
   * Trains the recommendation model on the provided interaction data.
   * Uses matrix factorization to learn user and entity embeddings.
   * Processes data in batches to reduce memory usage.
   * @param data - Array of user-entity interactions with optional ratings
   * @returns Promise that resolves when training is complete
   */
  async fit(data: Interaction[] = []) {
    if (!data.length) return;

    // Process user and entity registration in batches
    await this.registerUsersAndEntitiesInBatches(data);

    if (!this._initialized) {
      await this.initializeEmbeddings();
    } else {
      // Handle embedding resizing for new users/entities
      await this.resizeEmbeddingsIfNeeded();
    }

    // Train in batches for each epoch
    for (let e = 0; e < this.epoch; e++) {
      await this.trainEpochInBatches(data);
    }
  }

  /**
   * Registers users and entities from interaction data in batches.
   * @param data - Array of user-entity interactions
   * @private
   */
  private async registerUsersAndEntitiesInBatches(data: Interaction[]) {
    for (let i = 0; i < data.length; i += this.batchSize) {
      const batch = data.slice(i, i + this.batchSize);

      for (const { user, entity } of batch) {
        await this.storage.getOrCreateUserIndex(user);
        await this.storage.getOrCreateEntityIndex(entity);
      }

      // Add a small delay to prevent blocking the event loop
      if (i > 0 && i % (this.batchSize * 10) === 0) {
        await new Promise((resolve) => setTimeout(resolve, 1));
      }
    }
  }

  /**
   * Resizes embeddings if new users or entities have been added.
   * @private
   */
  private async resizeEmbeddingsIfNeeded() {
    const userCount = (await this.storage.getAllUsers()).length;
    const entityCount = (await this.storage.getAllEntities()).length;

    if (userCount > this.userEmbeddings.shape[0]) {
      const newUserCount = userCount - this.userEmbeddings.shape[0];
      const newUsers = await this.createEmbeddingsInBatches(
        newUserCount,
        this.embeddingSize
      );
      const oldUserEmbeddings = this.userEmbeddings;
      this.userEmbeddings = tf.variable(
        tf.concat([oldUserEmbeddings, newUsers], 0)
      );
      oldUserEmbeddings.dispose();
      newUsers.dispose();
    }

    if (entityCount > this.entityEmbeddings.shape[0]) {
      const newEntityCount = entityCount - this.entityEmbeddings.shape[0];
      const newEntities = await this.createEmbeddingsInBatches(
        newEntityCount,
        this.embeddingSize
      );
      const oldEntityEmbeddings = this.entityEmbeddings;
      this.entityEmbeddings = tf.variable(
        tf.concat([oldEntityEmbeddings, newEntities], 0)
      );
      oldEntityEmbeddings.dispose();
      newEntities.dispose();
    }
  }

  /**
   * Trains one epoch of the model using batch processing.
   * @param data - Array of user-entity interactions
   * @private
   */
  private async trainEpochInBatches(data: Interaction[]) {
    for (let i = 0; i < data.length; i += this.batchSize) {
      const batch = data.slice(i, i + this.batchSize);

      for (const { user, entity, rating } of batch) {
        const userIdx = await this.storage.getUserIndex(user);
        const entityIdx = await this.storage.getEntityIndex(entity);
        if (userIdx == null || entityIdx == null) continue;

        await this.optimizer.minimize(() => {
          const userVec = this.userEmbeddings.slice(
            [userIdx, 0],
            [1, this.embeddingSize]
          );
          const entityVec = this.entityEmbeddings.slice(
            [entityIdx, 0],
            [1, this.embeddingSize]
          );
          const pred = tf.sum(tf.mul(userVec, entityVec));
          const target = tf.scalar(rating ?? 1);
          return tf.losses.meanSquaredError(target, pred);
        }, true);
      }

      // Add a small delay every few batches to prevent blocking the event loop
      if (i > 0 && i % (this.batchSize * 5) === 0) {
        await new Promise((resolve) => setTimeout(resolve, 1));
      }
    }
  }

  /**
   * Gets the top users most likely to interact with a given entity.
   * @param entity - The entity ID to find users for
   * @param count - Maximum number of users to return (default: 10)
   * @returns Promise resolving to an array of user IDs ranked by likelihood
   */
  async getUsers(entity: Id, count = 10): Promise<Id[]> {
    const entityIdx = await this.storage.getEntityIndex(entity);
    if (entityIdx == null) return [];

    const entityVec = this.entityEmbeddings.slice(
      [entityIdx, 0],
      [1, this.embeddingSize]
    );
    const scores = tf
      .matMul(this.userEmbeddings, entityVec.transpose())
      .reshape([-1]);
    const values = (await scores.array()) as number[];
    const userIds = await this.storage.getAllUsers();

    return this.rankTop(values, userIds, count);
  }

  /**
   * Gets the top entities most likely to be of interest to a given user.
   * @param user - The user ID to find recommendations for
   * @param count - Maximum number of entities to return (default: 10)
   * @returns Promise resolving to an array of entity IDs ranked by likelihood
   */
  async getEntities(user: Id, count = 10): Promise<Id[]> {
    const userIdx = await this.storage.getUserIndex(user);
    if (userIdx == null) return [];

    const userVec = this.userEmbeddings.slice(
      [userIdx, 0],
      [1, this.embeddingSize]
    );
    const scores = tf
      .matMul(this.entityEmbeddings, userVec.transpose())
      .reshape([-1]);
    const values = (await scores.array()) as number[];
    const entityIds = await this.storage.getAllEntities();

    return this.rankTop(values, entityIds, count);
  }

  /**
   * Ranks and returns the top items based on their scores.
   * @param scores - Array of numerical scores
   * @param ids - Array of corresponding IDs
   * @param count - Number of top items to return
   * @returns Array of top-ranked IDs
   * @private
   */
  private rankTop(scores: number[], ids: Id[], count: number): Id[] {
    return scores
      .map((score, idx) => ({ id: ids[idx], score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, count)
      .map((entry) => entry.id);
  }

  /**
   * Exports the trained model as a JSON-serializable object.
   * @returns Promise resolving to the serialized model data
   */
  async export(): Promise<SerializedModel> {
    if (!this._initialized) {
      throw new Error(
        'Model must be initialized before export. Call fit() first.'
      );
    }

    const storageData = await this.storage.exportData();
    const userEmbeddingsArray =
      (await this.userEmbeddings.array()) as number[][];
    const entityEmbeddingsArray =
      (await this.entityEmbeddings.array()) as number[][];

    return {
      config: {
        epoch: this.epoch,
        embeddingSize: this.embeddingSize,
        learningRate: this.learningRate,
        batchSize: this.batchSize,
      },
      userEmbeddings: userEmbeddingsArray,
      entityEmbeddings: entityEmbeddingsArray,
      userMap: storageData.userMap,
      entityMap: storageData.entityMap,
      reverseUserMap: storageData.reverseUserMap,
      reverseEntityMap: storageData.reverseEntityMap,
      initialized: this._initialized,
      version: '1.0.0',
    };
  }

  /**
   * Saves the trained model to a JSON file.
   * Note: This method requires Node.js environment with file system access.
   * @param filePath - Path where to save the model file
   * @returns Promise that resolves when the file is saved
   */
  async save(filePath: string): Promise<void> {
    // Dynamic import to handle environments without fs module
    try {
      const fs = await import('fs');
      const path = await import('path');

      const modelData = await this.export();
      const jsonString = JSON.stringify(modelData);

      // Ensure directory exists
      const dir = path.dirname(filePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }

      fs.writeFileSync(filePath, jsonString, 'utf8');
    } catch (error) {
      throw new Error(
        `Failed to save model: ${
          error instanceof Error ? error.message : 'Unknown error'
        }. Make sure you're running in a Node.js environment.`
      );
    }
  }

  /**
   * Creates a new Recommendation instance from a serialized model.
   * @param modelData - The serialized model data
   * @param storage - Optional custom storage strategy (defaults to InMemoryStorage)
   * @returns Promise resolving to a new Recommendation instance
   */
  static async from(
    modelData: SerializedModel,
    storage?: StorageStrategy
  ): Promise<Recommendation> {
    // Validate version compatibility
    if (modelData.version !== '1.0.0') {
      console.warn(
        `Loading model with version ${modelData.version}, expected 1.0.0. Some features may not work correctly.`
      );
    }

    const recommendation = new Recommendation({
      ...modelData.config,
      storage: storage ?? new InMemoryStorage(),
    });

    // Import storage data
    await recommendation.storage.importData({
      userMap: modelData.userMap,
      entityMap: modelData.entityMap,
      reverseUserMap: modelData.reverseUserMap,
      reverseEntityMap: modelData.reverseEntityMap,
    });

    // Restore embeddings
    recommendation.userEmbeddings = tf.variable(
      tf.tensor2d(modelData.userEmbeddings)
    );
    recommendation.entityEmbeddings = tf.variable(
      tf.tensor2d(modelData.entityEmbeddings)
    );
    recommendation._initialized = modelData.initialized;

    return recommendation;
  }

  /**
   * Creates a new Recommendation instance from a saved model file.
   * Note: This method requires Node.js environment with file system access.
   * @param filePath - Path to the saved model file
   * @param storage - Optional custom storage strategy (defaults to InMemoryStorage)
   * @returns Promise resolving to a new Recommendation instance
   */
  static async fromPath(
    filePath: string,
    storage?: StorageStrategy
  ): Promise<Recommendation> {
    try {
      const fs = await import('fs');

      if (!fs.existsSync(filePath)) {
        throw new Error(`Model file not found: ${filePath}`);
      }

      const jsonString = fs.readFileSync(filePath, 'utf8');
      const modelData: SerializedModel = JSON.parse(jsonString);

      return this.from(modelData, storage);
    } catch (error) {
      throw new Error(
        `Failed to load model: ${
          error instanceof Error ? error.message : 'Unknown error'
        }. Make sure you're running in a Node.js environment and the file exists.`
      );
    }
  }
}
