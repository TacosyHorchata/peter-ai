// memeory manager

import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAI } from 'openai';
import { v4 as uuidv4 } from 'uuid';

function calculateCosineSimilarity(vecA: number[], vecB: number[]): number {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * (vecB[i] || 0), 0);
    const normA = Math.sqrt(vecA.reduce((sum, a) => sum + (a * a), 0));
    const normB = Math.sqrt(vecB.reduce((sum, b) => sum + (b * b), 0));
    return (normA === 0 || normB === 0) ? 0 : dotProduct / (normA * normB);
}

interface Memory {
    id: string;
    content: string;
    embedding: number[];
    metadata: {
        timestamp: number;
        type: string;
        summary: string;
        importance: number;
        lastAccessed: number;
        version: number;
        // General flag to indicate a salient (key) memory.
        salient?: boolean;
        tags?: string[];
        relations?: string[]; // IDs of related memories for complex relationships
    };
}

/**
 * MemoryManager - A state-of-the-art, general memory management system that
 * supports full conversation histories, salient memories, consolidation of
 * related memories, conflict resolution (e.g. name facts), and complex semantic searches.
 */
export class MemoryManager {
    private pinecone: Pinecone;
    private openai: OpenAI;
    private indexName: string = 'personal-assistant';
    private namespace: string = 'memories';

    constructor() {
        if (!process.env.PINECONE_API_KEY) {
            throw new Error('PINECONE_API_KEY is not set');
        }
        if (!process.env.OPENAI_API_KEY) {
            throw new Error('OPENAI_API_KEY is not set');
        }

        this.pinecone = new Pinecone();
        this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    }

    // Generates an embedding vector for the provided text.
    private async generateEmbedding(text: string): Promise<number[]> {
        const response = await this.openai.embeddings.create({
            model: "text-embedding-3-small",
            input: text,
        });
        return response.data[0].embedding;
    }

    // Generates a concise summary for given text.
    private async generateSummary(text: string): Promise<string> {
        const response = await this.openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                {
                    role: "system",
                    content: "Summarize the following text concisely in under 100 words, highlighting key points."
                },
                {
                    role: "user",
                    content: text
                }
            ]
        });
        return response.choices[0].message.content || '';
    }

    // Evaluate the importance of the content on a scale from 0 to 1.
    private async evaluateImportance(content: string): Promise<number> {
        const response = await this.openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                {
                    role: "system",
                    content: "Rate the importance of the following content on a scale from 0 (trivial) to 1 (critical) and respond with only the number."
                },
                { role: "user", content }
            ]
        });
        return parseFloat(response.choices[0].message.content || '0.5');
    }

    // Evaluate if the content contains a salient fact.
    private async evaluateSalientMemory(content: string): Promise<{ isSalient: boolean; summary: string }> {
        const prompt = `
You are a memory classification assistant that identifies key, memorable information.
Given the following conversation:
"${content}"

Determine if it contains an important fact that should be remembered long term, such as:
- Personal information (names, preferences, relationships)
- Important dates or events
- Key decisions or agreements
- Significant preferences or dislikes
- Notable achievements or experiences

If yes, respond with a JSON object:
{"isSalient": true, "summary": "A clear, specific statement of the key fact (e.g., 'User's name is John', 'User prefers vegetarian food')"}

If not (if it's just casual conversation or non-essential information), respond with:
{"isSalient": false, "summary": ""}

Respond ONLY with the JSON object.
`;

        const response = await this.openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [{ role: "system", content: prompt }]
        });
        try {
            const jsonString = response.choices[0].message.content?.trim();
            return JSON.parse(jsonString as string);
        } catch (error) {
            return { isSalient: false, summary: "" };
        }
    }

    /**
     * Uses the LLM to decide if new information should update an existing salient memory.
     * This method is fully generalized – it compares the new input to the existing memory text and,
     * if they conflict or if the new information is more up-to-date, returns an update directive.
     *
     * @param newContent The new fact/information.
     * @param existingContent The previously stored fact/information.
     */
    private async evaluateMemoryUpdate(
        newContent: string,
        existingContent: string
    ): Promise<{ update: boolean; updatedSummary: string }> {
        const prompt = `
You are a memory management assistant that handles conflicts in a human-like memory system.
Existing memory: "${existingContent}"
New information: "${newContent}"
The new information might contradict, clarify, or update the existing fact.
Humans update their memories when confronted with clearer or more recent information.
If the new information contradicts or provides a more accurate version of the fact, respond with:
{"update": true, "updatedSummary": "<a concise updated fact merging the new and existing information>"}
If the new information is redundant or confirms the existing memory, respond with:
{"update": false, "updatedSummary": ""}
Respond ONLY with the JSON object.
`;
        const response = await this.openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [{ role: "system", content: prompt }]
        });
        try {
            const jsonString = response.choices[0].message.content?.trim();
            return JSON.parse(jsonString as string);
        } catch (error) {
            return { update: false, updatedSummary: "" };
        }
    }

    // Determines if the given text is trivial.
    private isTrivial(text: string): boolean {
        const message = text.split(':').pop()?.trim().toLowerCase() || text;
        const greetings = ["hey", "hi", "hello", "good morning", "good evening", "greetings"];
        if (greetings.includes(message)) return true;
        if (message.length < 5) return true;
        return false;
    }

    /**
     * Adds a new memory to storage.
     * It handles salient memory detection, generic conflict resolution via evaluateMemoryUpdate,
     * and then either updates an existing memory or creates a new one.
     *
     * @param content The full text content of the memory.
     * @param type Classification type (e.g., "conversation").
     * @param tags Optional array of tags.
     * @param relatedIds Optional array of related memory IDs.
     */
    async addMemory(
        content: string,
        type: string,
        tags: string[] = [],
        relatedIds: string[] = []
    ): Promise<void> {
        try {
            if (this.isTrivial(content)) {
                console.log("Content is trivial. Skipping memory creation.");
                return;
            }

            // First, evaluate if this is a salient memory
            const salientDecision = await this.evaluateSalientMemory(content);
            
            // If it's not salient and not explicitly marked as important, skip storage
            if (!salientDecision.isSalient && type !== 'important') {
                console.log("Content is not salient. Skipping memory creation.");
                return;
            }

            const index = this.pinecone.index(this.indexName);
            const embedding = await this.generateEmbedding(content);
            const importance = await this.evaluateImportance(content);

            // For salient memories, check if we already have a similar one
            if (salientDecision.isSalient) {
                const queryResponse = await index.query({
                    vector: embedding,
                    topK: 1,
                    includeMetadata: true,
                    filter: { salient: true }
                });

                if (queryResponse.matches && queryResponse.matches.length > 0) {
                    const existingMemory = queryResponse.matches[0];
                    const similarity = calculateCosineSimilarity(embedding, existingMemory.values);
                    
                    if (similarity > 0.8) {
                        console.log(`Found similar salient memory (${existingMemory.id}). Evaluating update...`);
                        const decision = await this.evaluateMemoryUpdate(
                            content,
                            existingMemory.metadata?.content as string
                        );
                        if (decision.update) {
                            await this.editMemory(existingMemory.id, content, decision.updatedSummary, true);
                            return;
                        } else {
                            console.log("Memory is similar but doesn't require an update. Skipping creation.");
                            return;
                        }
                    }
                }
            }

            // Create new memory with the salient summary if available
            console.log("Creating new salient memory...");
            const memoryId = uuidv4();
            const summary = salientDecision.isSalient ? 
                salientDecision.summary : 
                await this.generateSummary(content);

            const records = [{
                id: memoryId,
                values: embedding,
                metadata: {
                    content,
                    tags,
                    relations: relatedIds,
                    timestamp: Date.now(),
                    type,
                    summary,
                    importance,
                    lastAccessed: Date.now(),
                    version: 1,
                    salient: true  // Always mark as salient since we're only storing important memories
                }
            }];

            await index.upsert(records);
            console.log("Salient memory stored successfully.");
        } catch (error) {
            console.error("Error in addMemory:", error);
            throw error;
        }
    }

    /**
     * Edits an existing memory.
     *
     * @param memoryId The ID of the memory.
     * @param newContent Updated content.
     * @param newSummary (Optional) A new summary.
     * @param isSalient (Optional) Whether this should be marked as salient.
     */
    async editMemory(
        memoryId: string,
        newContent: string,
        newSummary?: string,
        isSalient: boolean = false
    ): Promise<void> {
        const index = this.pinecone.index(this.indexName);
        const updatedEmbedding = await this.generateEmbedding(newContent);
        const summary = newSummary || (await this.generateSummary(newContent));
        const importance = await this.evaluateImportance(newContent);
        const records = [
            {
                id: memoryId,
                values: updatedEmbedding,
                metadata: {
                    content: newContent,
                    timestamp: Date.now(),
                    type: "conversation",
                    summary,
                    importance,
                    lastAccessed: Date.now(),
                    version: 1,
                    ...(isSalient ? { salient: true } : {})
                }
            }
        ];

        // Directly upsert the records.
        await index.upsert(records);
    }

    /**
     * Retrieves memories related to the query using semantic similarity.
     *
     * @param query A query string.
     * @param limit Maximum number of memories.
     */
    async getRelatedMemories(query: string, limit: number = 5): Promise<Memory[]> {
        try {
            const index = this.pinecone.index(this.indexName);
            const queryEmbedding = await this.generateEmbedding(query);
            
            // Add filter to only search through salient memories
            const queryResponse = await (index.query as any)({
                vector: queryEmbedding,
                topK: limit,
                includeMetadata: true,
                filter: { salient: true }  // Only search through salient memories
            });

            console.log("Found memories:", queryResponse.matches?.length || 0);
            console.log("Query response:", JSON.stringify(queryResponse, null, 2));

            const memories = (queryResponse.matches?.map((match: any) => ({
                id: match.id,
                content: match.metadata.content,
                embedding: match.values,  // Include the embedding values
                metadata: match.metadata,
            })) || []);

            // Sort by similarity (most relevant first) and then by recency
            memories.sort((a: Memory, b: Memory) => {
                const similarityA = calculateCosineSimilarity(queryEmbedding, a.embedding);
                const similarityB = calculateCosineSimilarity(queryEmbedding, b.embedding);
                if (Math.abs(similarityA - similarityB) > 0.1) {
                    return similarityB - similarityA;
                }
                return b.metadata.lastAccessed - a.metadata.lastAccessed;
            });

            return memories.slice(0, limit);
        } catch (error) {
            console.error('Error in getRelatedMemories:', error);
            return [];
        }
    }

    /**
     * Retrieves memories using metadata filtering and semantic search.
     *
     * @param filter A key/value filter on metadata.
     * @param queryText A search term to generate the embedding.
     * @param limit Maximum number of memories.
     */
    async getMemoriesByFilter(filter: Record<string, any>, queryText: string, limit: number = 10): Promise<Memory[]> {
        try {
            const index = this.pinecone.index(this.indexName);
            const queryEmbedding = await this.generateEmbedding(queryText);
            const queryResponse = await (index.query as any)({
                vector: queryEmbedding,
                topK: limit * 2,
                includeMetadata: true,
                filter,
            });
            const memories = (queryResponse.matches?.map((match: any) => ({
                id: match.id,
                content: match.metadata.content,
                embedding: [],
                metadata: match.metadata,
            })) || []);
            memories.sort((a: Memory, b: Memory) => b.metadata.lastAccessed - a.metadata.lastAccessed);
            return memories.slice(0, limit);
        } catch (error) {
            console.error("Error in getMemoriesByFilter:", error);
            return [];
        }
    }

    /**
     * Performs an advanced memory search by combining semantic similarity and metadata filtering.
     *
     * @param queryText The query string.
     * @param filters Optional filters to refine search.
     * @param limit Maximum number of memories to return.
     */
    async searchMemoriesComplex(
        queryText: string,
        filters: Record<string, any> = {},
        limit: number = 5
    ): Promise<Memory[]> {
        try {
            const index = this.pinecone.index(this.indexName);
            const queryEmbedding = await this.generateEmbedding(queryText);
            console.log("Query embedding:", queryEmbedding);

            // Build query parameters without the namespace property.
            const queryParams: any = {
                vector: queryEmbedding,
                topK: limit,
                includeMetadata: true,
            };

            if (Object.keys(filters).length > 0) {
                queryParams.filter = filters;
            }

            const queryResponse = await (index.query as any)(queryParams);
            console.log("Pinecone query response:", queryResponse);

            const memories = (queryResponse.matches?.map((match: any) => ({
                id: match.id,
                content: match.metadata.content,
                embedding: [],
                metadata: match.metadata,
            })) || []);
            memories.sort((a: Memory, b: Memory) => b.metadata.lastAccessed - a.metadata.lastAccessed);
            return memories.slice(0, limit);
        } catch (error) {
            console.error("Error in searchMemoriesComplex:", error);
            return [];
        }
    }

    /**
     * Deletes a memory given its ID.
     *
     * @param memoryId The memory's ID.
     */
    async deleteMemory(memoryId: string): Promise<void> {
        const index = this.pinecone.index(this.indexName);
        await index.deleteMany([memoryId]);
    }

    /**
     * Consolidates a set of related memories (such as a long conversation thread)
     * into a single consolidated memory.
     *
     * @param threadMemories An array of memories from a conversation thread.
     */
    async consolidateThreadMemories(threadMemories: Memory[]): Promise<void> {
        if (threadMemories.length < 3) return; // Not enough memories to consolidate

        const consolidatedContent = await this.generateSummary(
            threadMemories.map(m => m.content).join('\n\n')
        );
        const consolidatedEmbedding = await this.generateEmbedding(consolidatedContent);
        const consolidatedMemoryId = uuidv4();

        const index = this.pinecone.index(this.indexName);
        const records = [
            {
                id: consolidatedMemoryId,
                values: consolidatedEmbedding,
                metadata: {
                    content: consolidatedContent,
                    timestamp: Date.now(),
                    type: "conversation_consolidated",
                    summary: consolidatedContent.slice(0, 200) + '...',
                    importance: 0.7,
                    lastAccessed: Date.now(),
                    version: 1,
                    tags: ["consolidated"],
                    relations: threadMemories.map(m => m.id)
                }
            }
        ];

        // Upsert direct records without a namespace.
        await index.upsert(records);
    }

    /**
     * Periodically resolves conflicts among all salient memories in a general fashion.
     * This method queries for salient memories and compares each pair using the LLM-driven updater.
     * If two memories conflict (i.e. one should be updated), it merges them.
     */
    async resolveMemoryConflicts(): Promise<void> {
        try {
            const index = this.pinecone.index(this.indexName);
            const queryResponse = await index.query({
                vector: Array(1536).fill(0), // placeholder vector
                topK: 5,
                includeMetadata: true,
                filter: { salient: true }
            });
            const memories = queryResponse.matches || [];
            for (let i = 0; i < memories.length; i++) {
                for (let j = i + 1; j < memories.length; j++) {
                    const memA = memories[i];
                    const memB = memories[j];
                    // Assume the more recent memory is more accurate.
                    if (memA.metadata?.lastAccessed && memB.metadata?.lastAccessed && memA.metadata?.lastAccessed >= memB.metadata?.lastAccessed) {
                        const decision = await this.evaluateMemoryUpdate(
                            memB.metadata?.content as string,
                            memA.metadata?.content as string
                        );
                        if (decision.update) {
                            console.log(`Merging memory ${memB.id} into ${memA.id}`);
                            await this.editMemory(memA.id, memA.metadata?.content as string, decision.updatedSummary, true);
                        }
                    } else {
                        const decision = await this.evaluateMemoryUpdate(
                            memA.metadata?.content as string,
                            memB.metadata?.content as string
                        );
                        if (decision.update) {
                            console.log(`Merging memory ${memA.id} into ${memB.id}`);
                            await this.editMemory(memB.id, memB.metadata?.content as string, decision.updatedSummary, true);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error in resolveMemoryConflicts:', error);
        }
    }
}

