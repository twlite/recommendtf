import { Recommendation } from '../src/recommendation';

const recommendation = new Recommendation({
  epoch: 3,
});

// Movie recommendation dataset with realistic user preferences
const data = [
  // Alice - loves sci-fi and action movies
  { user: 'alice_123', entity: 'interstellar_2014', rating: 5.0 },
  { user: 'alice_123', entity: 'blade_runner_2049', rating: 4.8 },
  { user: 'alice_123', entity: 'the_matrix_1999', rating: 4.9 },
  { user: 'alice_123', entity: 'inception_2010', rating: 4.7 },
  { user: 'alice_123', entity: 'romantic_comedy_2023', rating: 2.1 },
  { user: 'alice_123', entity: 'horror_movie_2022', rating: 1.5 },

  // Bob - enjoys comedies and dramas
  { user: 'bob_moviefan', entity: 'romantic_comedy_2023', rating: 4.5 },
  { user: 'bob_moviefan', entity: 'the_godfather_1972', rating: 5.0 },
  { user: 'bob_moviefan', entity: 'forrest_gump_1994', rating: 4.8 },
  { user: 'bob_moviefan', entity: 'the_matrix_1999', rating: 3.2 },
  { user: 'bob_moviefan', entity: 'horror_movie_2022', rating: 2.0 },

  // Carol - horror and thriller enthusiast
  { user: 'carol_cinephile', entity: 'horror_movie_2022', rating: 4.9 },
  {
    user: 'carol_cinephile',
    entity: 'psychological_thriller_2023',
    rating: 4.7,
  },
  { user: 'carol_cinephile', entity: 'inception_2010', rating: 4.3 },
  { user: 'carol_cinephile', entity: 'romantic_comedy_2023', rating: 1.8 },
  { user: 'carol_cinephile', entity: 'the_matrix_1999', rating: 3.5 },

  // David - classic movie lover
  { user: 'david_retro', entity: 'the_godfather_1972', rating: 5.0 },
  { user: 'david_retro', entity: 'casablanca_1942', rating: 4.9 },
  { user: 'david_retro', entity: 'citizen_kane_1941', rating: 4.7 },
  { user: 'david_retro', entity: 'forrest_gump_1994', rating: 4.6 },
  { user: 'david_retro', entity: 'blade_runner_2049', rating: 3.0 },

  // Emma - diverse tastes, moderately rates everything
  { user: 'emma_casual', entity: 'interstellar_2014', rating: 4.2 },
  { user: 'emma_casual', entity: 'romantic_comedy_2023', rating: 3.8 },
  { user: 'emma_casual', entity: 'horror_movie_2022', rating: 3.5 },
  { user: 'emma_casual', entity: 'the_godfather_1972', rating: 4.1 },
  { user: 'emma_casual', entity: 'forrest_gump_1994', rating: 4.0 },
  { user: 'emma_casual', entity: 'inception_2010', rating: 3.9 },

  // Frank - action movie enthusiast
  { user: 'frank_action', entity: 'the_matrix_1999', rating: 5.0 },
  { user: 'frank_action', entity: 'blade_runner_2049', rating: 4.5 },
  { user: 'frank_action', entity: 'inception_2010', rating: 4.8 },
  { user: 'frank_action', entity: 'mad_max_fury_road', rating: 4.9 },
  { user: 'frank_action', entity: 'romantic_comedy_2023', rating: 2.2 },
  { user: 'frank_action', entity: 'casablanca_1942', rating: 2.8 },

  // Grace - indie and art house films
  { user: 'grace_indie', entity: 'moonlight_2016', rating: 5.0 },
  { user: 'grace_indie', entity: 'her_2013', rating: 4.7 },
  { user: 'grace_indie', entity: 'citizen_kane_1941', rating: 4.5 },
  { user: 'grace_indie', entity: 'psychological_thriller_2023', rating: 4.2 },
  { user: 'grace_indie', entity: 'mad_max_fury_road', rating: 2.5 },
  { user: 'grace_indie', entity: 'horror_movie_2022', rating: 2.0 },

  // Henry - family-friendly content
  { user: 'henry_family', entity: 'toy_story_1995', rating: 4.9 },
  { user: 'henry_family', entity: 'finding_nemo_2003', rating: 4.8 },
  { user: 'henry_family', entity: 'romantic_comedy_2023', rating: 4.0 },
  { user: 'henry_family', entity: 'forrest_gump_1994', rating: 4.3 },
  { user: 'henry_family', entity: 'horror_movie_2022', rating: 1.2 },
  { user: 'henry_family', entity: 'psychological_thriller_2023', rating: 1.8 },

  // Ivy - documentary enthusiast
  { user: 'ivy_docs', entity: 'free_solo_2018', rating: 5.0 },
  { user: 'ivy_docs', entity: 'march_of_penguins_2005', rating: 4.6 },
  { user: 'ivy_docs', entity: 'citizen_kane_1941', rating: 4.4 },
  { user: 'ivy_docs', entity: 'interstellar_2014', rating: 3.8 },
  { user: 'ivy_docs', entity: 'romantic_comedy_2023', rating: 2.5 },
  { user: 'ivy_docs', entity: 'horror_movie_2022', rating: 1.5 },
];

async function main() {
  console.log('🎬 Training movie recommendation system...');
  await recommendation.fit(data);
  console.log('✅ Training completed!\n');

  // Get movie recommendations for Alice (sci-fi lover)
  console.log(
    '🔍 Getting movie recommendations for Alice (sci-fi enthusiast):'
  );
  const aliceRecommendations = await recommendation.getEntities('alice_123');
  console.log('Recommended movies for Alice:', aliceRecommendations);
  console.log('');

  // Get movie recommendations for Bob (comedy/drama fan)
  console.log('🔍 Getting movie recommendations for Bob (comedy/drama fan):');
  const bobRecommendations = await recommendation.getEntities('bob_moviefan');
  console.log('Recommended movies for Bob:', bobRecommendations);
  console.log('');

  // Find users similar to Carol (horror enthusiast)
  console.log('👥 Finding users with similar taste to Carol (horror fan):');
  const carolSimilarUsers = await recommendation.getUsers('carol_cinephile');
  console.log('Users similar to Carol:', carolSimilarUsers);
  console.log('');

  // Find users who might like "The Matrix"
  console.log('🎭 Finding users who might enjoy "The Matrix":');
  const matrixFans = await recommendation.getUsers('the_matrix_1999');
  console.log('Potential Matrix fans:', matrixFans);
  console.log('');

  // Find users who might like "Romantic Comedy 2023"
  console.log('💕 Finding users who might enjoy "Romantic Comedy 2023":');
  const romcomFans = await recommendation.getUsers('romantic_comedy_2023');
  console.log('Potential romantic comedy fans:', romcomFans);

  // train new data
  await recommendation.fit([
    { user: 'alice_123', entity: 'mad_max_fury_road', rating: 4.5 },
    { user: 'bob_moviefan', entity: 'mad_max_fury_road', rating: 4.0 },
    { user: 'carol_cinephile', entity: 'mad_max_fury_road', rating: 4.8 },
    { user: 'david_retro', entity: 'mad_max_fury_road', rating: 4.2 },
    { user: 'emma_casual', entity: 'mad_max_fury_road', rating: 3.9 },
  ]);

  console.log('✅ Additional training completed!\n');

  // Get updated recommendations for Alice after new training
  console.log('🔍 Getting updated movie recommendations for Alice:');
  const updatedAliceRecommendations = await recommendation.getEntities(
    'alice_123'
  );
  console.log(
    'Updated recommended movies for Alice:',
    updatedAliceRecommendations
  );
  console.log('');
  // Get updated recommendations for Bob after new training
  console.log('🔍 Getting updated movie recommendations for Bob:');
  const updatedBobRecommendations = await recommendation.getEntities(
    'bob_moviefan'
  );
  console.log('Updated recommended movies for Bob:', updatedBobRecommendations);
}

// Example function to demonstrate batch processing with large datasets
async function demonstrateBatchProcessing() {
  // Example showing batch processing for better memory management
  const recommendationWithBatching = await (async () => {
    try {
      const startTime = Date.now();
      const rec = Recommendation.fromPath('./examples/batch_model.json');
      const endTime = Date.now();
      console.log(
        `Loaded recommendation model from file in ${endTime - startTime}ms`
      );
      return rec;
    } catch {
      const rec = new Recommendation({
        epoch: 3,
        batchSize: 500, // Process data in batches of 500 interactions
      });

      // Generate a larger dataset for demonstration
      const largeData: Array<{ user: string; entity: string; rating: number }> =
        [];
      for (let i = 0; i < 5000; i++) {
        largeData.push({
          user: `user_${i % 1000}`, // 1000 unique users
          entity: `item_${i % 500}`, // 500 unique items
          rating: Math.random() * 5 + 1, // Random rating between 1-6
        });
      }

      console.log(
        `Training with ${largeData.length} interactions in batches...`
      );

      const startTime = Date.now();
      await rec.fit(largeData);
      const endTime = Date.now();

      console.log(`Training completed in ${endTime - startTime}ms`);

      return rec;
    }
  })();

  console.log('Demonstrating batch processing for large datasets...');

  // Get recommendations for a specific user
  const recommendations = await recommendationWithBatching.getEntities(
    'user_42',
    5
  );
  console.log('Top 5 recommendations for user_42:', recommendations);

  await recommendationWithBatching.save('./examples/batch_model.json');
}

// Example function to demonstrate save and load functionality
async function demonstrateSaveLoad() {
  console.log('\n=== Demonstrating Save/Load Functionality ===');

  // Train a model
  const originalModel = new Recommendation({
    epoch: 5,
    embeddingSize: 8,
    learningRate: 0.1,
    batchSize: 100,
  });

  console.log('Training original model...');
  await originalModel.fit(data);

  // Get some recommendations from the original model
  const originalRecommendations = await originalModel.getEntities(
    'alice_123',
    3
  );
  console.log(
    'Original model recommendations for alice_123:',
    originalRecommendations
  );

  // Export the model to JSON
  console.log('Exporting model to JSON...');
  const modelJson = await originalModel.export();
  console.log(
    'Model exported successfully. JSON size:',
    JSON.stringify(modelJson).length,
    'characters'
  );

  // Save the model to a file
  console.log('Saving model to file...');
  await originalModel.save('./saved_model.json');
  console.log('Model saved to ./saved_model.json');

  // Load the model from JSON
  console.log('Loading model from JSON...');
  const loadedFromJson = await Recommendation.from(modelJson);
  const jsonRecommendations = await loadedFromJson.getEntities('alice_123', 3);
  console.log(
    'JSON-loaded model recommendations for alice_123:',
    jsonRecommendations
  );

  // Load the model from file
  console.log('Loading model from file...');
  const loadedFromFile = await Recommendation.fromPath('./saved_model.json');
  const fileRecommendations = await loadedFromFile.getEntities('alice_123', 3);
  console.log(
    'File-loaded model recommendations for alice_123:',
    fileRecommendations
  );

  // Verify that all models produce the same results
  const resultsMatch =
    JSON.stringify(originalRecommendations) ===
      JSON.stringify(jsonRecommendations) &&
    JSON.stringify(jsonRecommendations) === JSON.stringify(fileRecommendations);

  console.log(
    'All models produce identical results:',
    resultsMatch ? '✅' : '❌'
  );

  if (resultsMatch) {
    console.log('Save/Load functionality working correctly!');
  } else {
    console.log('There may be an issue with the save/load implementation.');
  }
}

// Main function to run all examples
async function runAllExamples() {
  try {
    // Run the original example (if main function exists)
    if (typeof main === 'function') {
      // await main();
    }

    // Run batch processing example
    await demonstrateBatchProcessing();

    // Run save/load example
    // await demonstrateSaveLoad();
  } catch (error) {
    console.error('Error running examples:', error);
  }
}

// Run all examples
runAllExamples().catch(console.error);
