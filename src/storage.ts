/**
 * Represents a unique identifier that can be either a string or number.
 */
export type Id = string | number;

/**
 * Serializable storage data for saving/loading models.
 */
export interface SerializableStorageData {
  /** User ID to index mapping */
  userMap: Array<[Id, number]>;
  /** Entity ID to index mapping */
  entityMap: Array<[Id, number]>;
  /** Reverse user mapping (index to ID) */
  reverseUserMap: Id[];
  /** Reverse entity mapping (index to ID) */
  reverseEntityMap: Id[];
}

/**
 * Interface defining the contract for storage strategies used by the recommendation system.
 * Implementations handle the mapping between user/entity IDs and their internal indices.
 */
export interface StorageStrategy {
  /**
   * Retrieves the internal index for a given user ID.
   * @param id - The user ID to look up
   * @returns Promise resolving to the user's index, or null if not found
   */
  getUserIndex(id: Id): Promise<number | null>;

  /**
   * Gets the internal index for a user ID, creating a new one if it doesn't exist.
   * @param id - The user ID to look up or create
   * @returns Promise resolving to the user's index
   */
  getOrCreateUserIndex(id: Id): Promise<number>;

  /**
   * Retrieves the internal index for a given entity ID.
   * @param id - The entity ID to look up
   * @returns Promise resolving to the entity's index, or null if not found
   */
  getEntityIndex(id: Id): Promise<number | null>;

  /**
   * Gets the internal index for an entity ID, creating a new one if it doesn't exist.
   * @param id - The entity ID to look up or create
   * @returns Promise resolving to the entity's index
   */
  getOrCreateEntityIndex(id: Id): Promise<number>;

  /**
   * Retrieves all user IDs in the order of their internal indices.
   * @returns Promise resolving to an array of all user IDs
   */
  getAllUsers(): Promise<Id[]>;

  /**
   * Retrieves all entity IDs in the order of their internal indices.
   * @returns Promise resolving to an array of all entity IDs
   */
  getAllEntities(): Promise<Id[]>;

  /**
   * Exports the storage data for serialization.
   * @returns Promise resolving to serializable storage data
   */
  exportData(): Promise<SerializableStorageData>;

  /**
   * Imports storage data from serialized format.
   * @param data - The serializable storage data to import
   * @returns Promise that resolves when import is complete
   */
  importData(data: SerializableStorageData): Promise<void>;
}
