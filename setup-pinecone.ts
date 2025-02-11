import { Pinecone } from '@pinecone-database/pinecone';
import * as dotenv from 'dotenv';

dotenv.config();

async function setupPinecone() {
    try {
        const pc = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY!
        });

        // Check if index already exists
        const indexList = await pc.listIndexes();
        if (indexList.indexes?.some((index: { name: string }) => index.name === 'personal-assistant')) {
            console.log('Index already exists, skipping creation');
            return;
        }

        // Create index for embeddings
        // Note: dimension should match your embedding model
        // OpenAI's text-embedding-3-small has 1536 dimensions
        await pc.createIndex({
            name: 'personal-assistant',
            dimension: 1536,
            metric: 'cosine',
            spec: {
                serverless: {
                    cloud: 'aws',
                    region: 'us-east-1'
                }
            }
        });

        console.log('Pinecone index created successfully');
    } catch (error) {
        console.error('Error setting up Pinecone:', error);
        throw error;
    }
}

setupPinecone().catch(console.error);