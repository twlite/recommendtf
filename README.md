# recommendtf

A powerful TensorFlow.js-based collaborative filtering recommendation system for Node.js. Built with TypeScript and designed for performance, scalability, and ease of use.

## Features

- 🧠 **Matrix Factorization**: Uses embeddings-based collaborative filtering for accurate recommendations
- ⚡ **Performance Optimized**: Batch processing to handle large datasets efficiently
- 💾 **Flexible Storage**: Pluggable storage strategies (in-memory included, custom implementations supported)
- 📊 **Scalable**: Handles dynamic user and entity addition without retraining from scratch
- 📝 **TypeScript**: Full type safety and excellent IDE support

## Installation

```bash
npm install recommendtf
```

You also need to install TensorFlow.js:

```bash
npm install @tensorflow/tfjs
```

## Quick Start

```typescript
import { Recommendation } from 'recommendtf';

// Create a new recommendation system
const recommender = new Recommendation({
  epoch: 5, // Training epochs
  embeddingSize: 16, // Embedding dimensions
  learningRate: 0.05, // Learning rate
  batchSize: 1000, // Batch size for processing
});

// Training data: user-entity interactions
const interactions = [
  { user: 'user1', entity: 'movie1', rating: 5.0 },
  { user: 'user1', entity: 'movie2', rating: 4.5 },
  { user: 'user2', entity: 'movie1', rating: 4.0 },
  { user: 'user2', entity: 'movie3', rating: 4.8 },
  // ... more interactions
];

// Train the model
await recommender.fit(interactions);

// Get recommendations for a user
const recommendations = await recommender.getEntities('user1', 5);
console.log('Recommended entities:', recommendations);

// Find users likely to be interested in an entity
const interestedUsers = await recommender.getUsers('movie1', 5);
console.log('Interested users:', interestedUsers);
```

## API Reference

### Class: `Recommendation`

Main class for the recommendation system.

#### Constructor

```typescript
new Recommendation(options?: RecommendationOptions)
```

**Options:**

- `epoch?: number` - Number of training epochs (default: 5)
- `embeddingSize?: number` - Size of embedding vectors (default: 16)
- `learningRate?: number` - Learning rate for optimizer (default: 0.05)
- `batchSize?: number` - Batch size for processing (default: 1000)
- `storage?: StorageStrategy` - Storage strategy (default: InMemoryStorage)
- `disableIsNode?: boolean` - Disable IS_NODE variable (default: false)

The `disableIsNode: true` will suppress the following warning when running in Node.js:

```
============================
Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details.
============================
```

#### Methods

##### `fit(data: Interaction[]): Promise<void>`

Trains the recommendation model on interaction data.

**Parameters:**

- `data` - Array of user-entity interactions

**Interaction Format:**

```typescript
interface Interaction {
  user: string | number; // User identifier
  entity: string | number; // Entity identifier
  rating?: number; // Optional rating (defaults to 1)
}
```

Calling this method multiple times allows for incremental learning, where new interactions are added without losing previous knowledge.

##### `getEntities(user: Id, count?: number): Promise<Id[]>`

Gets recommended entities for a user.

**Parameters:**

- `user` - User identifier
- `count` - Number of recommendations to return (default: 10)

**Returns:** Array of entity IDs ranked by recommendation score

##### `getUsers(entity: Id, count?: number): Promise<Id[]>`

Gets users most likely to be interested in an entity.

**Parameters:**

- `entity` - Entity identifier
- `count` - Number of users to return (default: 10)

**Returns:** Array of user IDs ranked by interest likelihood

##### `export(): Promise<SerializedModel>`

Exports the trained model as a JSON-serializable object.

**Returns:** Promise resolving to serialized model data

##### `save(filePath: string): Promise<void>`

Saves the trained model to a JSON file (Node.js only).

**Parameters:**

- `filePath` - Path where to save the model

##### `static from(modelData: SerializedModel, storage?: StorageStrategy): Promise<Recommendation>`

Creates a new instance from serialized model data.

**Parameters:**

- `modelData` - Serialized model data
- `storage` - Optional custom storage strategy

##### `static fromPath(filePath: string, storage?: StorageStrategy): Promise<Recommendation>`

Loads a model from a saved JSON file (Node.js only).

**Parameters:**

- `filePath` - Path to the saved model file
- `storage` - Optional custom storage strategy

## Storage Strategies

The library supports pluggable storage strategies for managing user and entity mappings.

### InMemoryStorage (Default)

Stores all mappings in memory. Suitable for development and small to medium datasets.

```typescript
import { InMemoryStorage } from 'recommendtf';

const storage = new InMemoryStorage();
const recommender = new Recommendation({ storage });
```

### Custom Storage

Implement the `StorageStrategy` interface for custom storage solutions:

```typescript
import { StorageStrategy, Id, SerializableStorageData } from 'recommendtf';

class CustomStorage implements StorageStrategy {
  async getUserIndex(id: Id): Promise<number | null> {
    // Your implementation
  }

  async getOrCreateUserIndex(id: Id): Promise<number> {
    // Your implementation
  }

  // ... implement all required methods
}
```

## Examples

### Movie Recommendation System

```typescript
import { Recommendation } from 'recommendtf';

const recommender = new Recommendation({
  epoch: 10,
  embeddingSize: 32,
  learningRate: 0.01,
});

// Movie ratings dataset
const movieRatings = [
  { user: 'alice', entity: 'inception', rating: 5.0 },
  { user: 'alice', entity: 'interstellar', rating: 4.8 },
  { user: 'bob', entity: 'comedy_movie', rating: 4.5 },
  { user: 'bob', entity: 'inception', rating: 3.2 },
  // ... more ratings
];

await recommender.fit(movieRatings);

// Get movie recommendations for Alice
const aliceRecommendations = await recommender.getEntities('alice', 5);
console.log('Movies for Alice:', aliceRecommendations);

// Find users who might like Inception
const inceptionFans = await recommender.getUsers('inception', 10);
console.log('Inception fans:', inceptionFans);

// Save the trained model
await recommender.save('./models/movie-recommender.json');
```

### E-commerce Product Recommendations

```typescript
// Load a pre-trained model
const recommender = await Recommendation.fromPath(
  './models/product-recommender.json'
);

// Get product recommendations for a customer
const productRecommendations = await recommender.getEntities(
  'customer_123',
  10
);

// Find customers interested in a specific product
const interestedCustomers = await recommender.getUsers('product_456', 20);
```

### Incremental Learning

```typescript
const recommender = new Recommendation();

// Initial training
await recommender.fit(initialData);

// Later, add more data without losing previous learning
await recommender.fit(newInteractions);

// The model automatically handles new users and entities
const recommendations = await recommender.getEntities('new_user', 5);
```

## Performance Considerations

### Batch Processing

The library automatically processes data in batches to optimize memory usage and prevent blocking the event loop:

- **Training**: Data is processed in configurable batch sizes
- **Embedding Creation**: Large embedding matrices are created incrementally
- **Memory Management**: Intermediate tensors are properly disposed

### Configuration Tips

```typescript
// For large datasets
const recommender = new Recommendation({
  batchSize: 5000, // Larger batches for better performance
  embeddingSize: 64, // Higher dimensional embeddings
  epoch: 20, // More training epochs
});

// For memory-constrained environments
const recommender = new Recommendation({
  batchSize: 500, // Smaller batches
  embeddingSize: 8, // Lower dimensional embeddings
  epoch: 5, // Fewer epochs
});
```

## Model Persistence

The library currently supports json export/import only.

### JSON Format (Human-readable)

```typescript
// Save to JSON file (larger file size)
await recommender.save('./models/my-model.json');

// Load from JSON file
const recommender = await Recommendation.fromPath('./models/my-model.json');

// Export/import as object (for in-memory operations)
const modelData = await recommender.export();
// Store modelData in database, cache, or other storage
const recommender = await Recommendation.from(modelData);
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support, please open an issue on the GitHub repository.
