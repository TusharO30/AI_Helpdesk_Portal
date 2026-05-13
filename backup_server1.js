require('dotenv').config();
const dns = require('dns'); 
dns.setServers(['8.8.8.8', '8.8.4.4']); 

const express = require('express');
const cors = require('cors');
const { GoogleGenAI } = require('@google/genai');
const mongoose = require('mongoose');
const multer = require('multer');
const { PDFParse } = require('pdf-parse');

const app = express();
app.use(cors());
app.use(express.json());

const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
});

// Set up multer to store uploaded files in memory temporarily
const upload = multer({ storage: multer.memoryStorage() });

// Helper Function: Chop long text into smaller chunks
function chunkText(text, chunkSize = 1000) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.substring(i, i + chunkSize));
    }
    return chunks;
}

// ==========================================
// 1. DATABASE CONNECTION & MODELS
// ==========================================
mongoose.connect(process.env.MONGO_URI)
    .then(() => console.log("✅ Connected to MongoDB Atlas"))
    .catch(err => console.error("❌ MongoDB connection error:", err));

// Schema for our Documents
const knowledgeSchema = new mongoose.Schema({
    text: String,
    embedding: [Number] 
});
const Knowledge = mongoose.model('Knowledge', knowledgeSchema);

// Schema for our Tickets
const ticketSchema = new mongoose.Schema({
    question: String,
    status: { type: String, default: 'Pending' },
    createdAt: { type: Date, default: Date.now }
});
const Ticket = mongoose.model('Ticket', ticketSchema);


// ==========================================
// 2. UPLOAD TO DATABASE (TEXT)
// ==========================================
app.post('/upload-doc', async (req, res) => {
    const { text } = req.body;
    
    try {
        const response = await ai.models.embedContent({
            model: 'gemini-embedding-2',
            contents: text,
        });
        
        const embedding = response.embeddings[0].values;
        
        // Save permanently to MongoDB
        const newDoc = new Knowledge({ text, embedding });
        await newDoc.save();
        
        res.json({ message: "Document uploaded and vectorized to MongoDB successfully!" });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "Failed to embed document." });
    }
});

// ==========================================
// NEW FEATURE: UPLOAD & CHUNK PDF
// ==========================================
app.post('/upload-pdf', upload.single('pdfDocument'), async (req, res) => {
    // 1. Check if file exists
    if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
    }

    try {
        // 2. Extract text from the PDF file
        const parser = new PDFParse({ data: req.file.buffer });
        const result = await parser.getText();
        const fullText = result.text;

        // 3. Chop the text into chunks of 1000 characters
        const chunks = chunkText(fullText, 1000); 
        let savedChunksCount = 0;

        // 4. Loop through each chunk, vectorize it, and save to MongoDB
        for (const chunk of chunks) {
            // Skip empty or really tiny chunks
            if (chunk.trim().length < 20) continue; 

            // Get embedding from Gemini
            const response = await ai.models.embedContent({
                model: 'gemini-embedding-2',
                contents: chunk,
            });
            const embedding = response.embeddings[0].values;

            // Save chunk to DB
            const newDoc = new Knowledge({ text: chunk, embedding });
            await newDoc.save();
            savedChunksCount++;
        }

        res.json({ message: `PDF Processed! Saved ${savedChunksCount} knowledge chunks to the AI Brain.` });

    } catch (error) {
        console.error("PDF Processing Error:", error);
        res.status(500).json({ error: "Failed to process PDF." });
    }
});

// ==========================================
// 3. ASK & DATABASE VECTOR SEARCH (UPDATED)
// ==========================================
app.post('/ask', async (req, res) => {
    const { question, history = [] } = req.body;
    
    try {
        // 1. Generate Embedding for the question
        const questionEmbeddingRes = await ai.models.embedContent({
            model: 'gemini-embedding-2', // Ensure this matches your upload model
            contents: question,
        });
        const questionEmbedding = questionEmbeddingRes.embeddings[0].values;

        // 2. Perform Vector Search in MongoDB
        const results = await Knowledge.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index", 
                    "path": "embedding",
                    "queryVector": questionEmbedding,
                    "numCandidates": 10,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "text": 1,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ]);
        
        console.log(`🧠 Found ${results.length} chunks. Top score: ${results[0]?.score}`);

        // 3. Score Threshold Check
        // If the best match is weak (under 0.7), create a ticket immediately
        if (results.length === 0 || results[0].score < 0.4) {
            const newTicket = new Ticket({ question });
            await newTicket.save();
            
            return res.json({ 
                answer: "I'm not confident about this answer. A ticket has been created for human support.",
                ticket: newTicket
            });
        }

        // 4. Prepare Context and AI Prompt
        const combinedContext = results.map(doc => doc.text).join('\n\n...\n\n');
        const formattedHistory = history.map(msg => ({
            role: msg.role === 'user' ? 'user' : 'model',
            parts: [{ text: msg.text }]
        }));

        formattedHistory.push({
            role: 'user',
            parts: [{ text: `Context: ${combinedContext}\n\nQuestion: ${question}` }]
        });

        // 5. Generate AI Response
        const aiResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash', 
            contents: formattedHistory,
            config: {
                systemInstruction: "You are a company helpdesk assistant. Answer the user's question based ONLY on the context provided and by identifying relevant information even it requires outside knowledge. Keep it brief. Always end your response by stating exactly where you found the information. If the answer is not in the context, start your response with 'NOT_FOUND'."
            }
        });

        // Define the variable correctly from the AI response
        const finalAnswer = aiResponse.text; 

        // 6. Final "I don't know" Check
        if (finalAnswer.includes("NOT_FOUND")) {
            const newTicket = new Ticket({ question });
            await newTicket.save();
            return res.json({
                answer: "The information isn't in our policy documents. I've created a support ticket for you.",
                ticket: newTicket
            });
        }

        // Success: Return AI answer and source snippets
        res.json({ 
            answer: finalAnswer, 
            sources: results.map(doc => doc.text) 
        });

    } catch (error) {
        console.error("Vector Search Error:", error);
        res.status(500).json({ error: "Failed to generate answer." });
    }
});
// ==========================================
// 4. TICKETING SYSTEM (ADMIN CONTROLS)
// ==========================================

// Fetch all unresolved tickets
app.get('/tickets', async (req, res) => {
    try {
        // Find tickets where status is Pending, sort by newest first
        const tickets = await Ticket.find({ status: 'Pending' }).sort({ createdAt: -1 });
        res.json(tickets);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "Failed to fetch tickets." });
    }
});

// Mark a ticket as resolved
// Updated Resolve Ticket Route with Auto-Injection
app.post('/resolve-ticket', async (req, res) => {
    const { ticketId, adminAnswer } = req.body; // Now accepting the actual answer
    
    try {
        // 1. Find the original ticket to get the question
        const ticket = await Ticket.findById(ticketId);
        if (!ticket) return res.status(404).json({ error: "Ticket not found" });

        // 2. Vectorize the Admin's Answer
        const response = await ai.models.embedContent({
            model: 'gemini-embedding-2',
            contents: `Question: ${ticket.question} | Answer: ${adminAnswer}`,
        });
        const embedding = response.embeddings[0].values;

        // 3. Save this new knowledge to the Knowledge collection
        const newKnowledge = new Knowledge({
            text: `Q: ${ticket.question} A: ${adminAnswer}`,
            embedding: embedding
        });
        await newKnowledge.save();

        // 4. Mark ticket as resolved
        ticket.status = 'Resolved';
        await ticket.save();

        res.json({ message: "Knowledge injected and ticket resolved!" });
    } catch (error) {
        console.error("Injection Error:", error);
        res.status(500).json({ error: "Failed to inject knowledge." });
    }
});
// ==========================================
// 5. LIBRARY MANAGEMENT (NEW)
// ==========================================

// Get all stored knowledge snippets
    app.get('/library', async (req, res) => {
        try {
            // We only fetch the text and ID, not the heavy embedding vectors
            const library = await Knowledge.find({}, { text: 1, _id: 1 });
            res.json(library);
        } catch (error) {
            res.status(500).json({ error: "Failed to fetch library." });
        }
    });

    // Delete a specific knowledge snippet
    app.delete('/library/:id', async (req, res) => {
        try {
            await Knowledge.findByIdAndDelete(req.params.id);
            res.json({ message: "Knowledge removed from AI Brain." });
        } catch (error) {
            res.status(500).json({ error: "Failed to delete knowledge." });
        }
    });
    // ==========================================
// 6. ANALYTICS DASHBOARD (NEW)
// ==========================================

        app.get('/analytics', async (req, res) => {
            try {
                const totalKnowledge = await Knowledge.countDocuments();
                const totalTickets = await Ticket.countDocuments();
                const pendingTickets = await Ticket.countDocuments({ status: 'Pending' });
                const resolvedTickets = await Ticket.countDocuments({ status: 'Resolved' });

                // Calculate AI Success Rate (Total requests that didn't become pending tickets)
                // Note: In a production app, you'd log every /ask request to a separate collection
                
                res.json({
                    knowledgeCount: totalKnowledge,
                    totalTickets: totalTickets,
                    pending: pendingTickets,
                    resolved: resolvedTickets
                });
            } catch (error) {
                res.status(500).json({ error: "Failed to fetch analytics." });
            }
        });

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Helpdesk MVP (MongoDB Edition) running on http://localhost:${PORT}`);
});