import { Id, StorageStrategy, SerializableStorageData } from './storage';

/**
 * In-memory implementation of the StorageStrategy interface.
 * Stores user and entity mappings in memory using Maps and arrays.
 * This implementation is suitable for development and small datasets.
 */
export class InMemoryStorage implements StorageStrategy {
  /** Maps user IDs to their internal indices */
  private userMap = new Map<Id, number>();
  /** Maps entity IDs to their internal indices */
  private entityMap = new Map<Id, number>();
  /** Array storing user IDs in index order for reverse lookup */
  private reverseUserMap: Id[] = [];
  /** Array storing entity IDs in index order for reverse lookup */
  private reverseEntityMap: Id[] = [];

  /**
   * Retrieves the internal index for a given user ID.
   * @param id - The user ID to look up
   * @returns Promise resolving to the user's index, or null if not found
   */
  async getUserIndex(id: Id): Promise<number | null> {
    return this.userMap.has(id) ? this.userMap.get(id)! : null;
  }

  /**
   * Gets the internal index for a user ID, creating a new one if it doesn't exist.
   * @param id - The user ID to look up or create
   * @returns Promise resolving to the user's index
   */
  async getOrCreateUserIndex(id: Id): Promise<number> {
    if (!this.userMap.has(id)) {
      const idx = this.userMap.size;
      this.userMap.set(id, idx);
      this.reverseUserMap[idx] = id;
    }
    return this.userMap.get(id)!;
  }

  /**
   * Retrieves the internal index for a given entity ID.
   * @param id - The entity ID to look up
   * @returns Promise resolving to the entity's index, or null if not found
   */
  async getEntityIndex(id: Id): Promise<number | null> {
    return this.entityMap.has(id) ? this.entityMap.get(id)! : null;
  }

  /**
   * Gets the internal index for an entity ID, creating a new one if it doesn't exist.
   * @param id - The entity ID to look up or create
   * @returns Promise resolving to the entity's index
   */
  async getOrCreateEntityIndex(id: Id): Promise<number> {
    if (!this.entityMap.has(id)) {
      const idx = this.entityMap.size;
      this.entityMap.set(id, idx);
      this.reverseEntityMap[idx] = id;
    }
    return this.entityMap.get(id)!;
  }

  /**
   * Retrieves all user IDs in the order of their internal indices.
   * @returns Promise resolving to an array of all user IDs
   */
  async getAllUsers(): Promise<Id[]> {
    return this.reverseUserMap;
  }

  /**
   * Retrieves all entity IDs in the order of their internal indices.
   * @returns Promise resolving to an array of all entity IDs
   */
  async getAllEntities(): Promise<Id[]> {
    return this.reverseEntityMap;
  }

  /**
   * Exports the storage data for serialization.
   * @returns Promise resolving to serializable storage data
   */
  async exportData(): Promise<SerializableStorageData> {
    return {
      userMap: Array.from(this.userMap.entries()),
      entityMap: Array.from(this.entityMap.entries()),
      reverseUserMap: [...this.reverseUserMap],
      reverseEntityMap: [...this.reverseEntityMap],
    };
  }

  /**
   * Imports storage data from serialized format.
   * @param data - The serializable storage data to import
   * @returns Promise that resolves when import is complete
   */
  async importData(data: SerializableStorageData): Promise<void> {
    this.userMap = new Map(data.userMap);
    this.entityMap = new Map(data.entityMap);
    this.reverseUserMap = [...data.reverseUserMap];
    this.reverseEntityMap = [...data.reverseEntityMap];
  }
}
