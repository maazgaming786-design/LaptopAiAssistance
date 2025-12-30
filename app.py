import os
import pandas as pd
import faiss
import numpy as np
import gradio as gr
import requests
import logging
from sentence_transformers import SentenceTransformer
from groq import Groq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read API Key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("Please set GROQ_API_KEY environment variable")

client = Groq(api_key=GROQ_API_KEY)

# Embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Vector Store Globals
index = None
documents = []
df = None

# Knowledge Base URL (Viewable CSV)
GDRIVE_FILE_ID = "1L7-R8wUUjGJ5QRYV3OzORhZPXi4TE1H_"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"


def build_knowledge_base():
    global index, documents, df
    
    try:
        # Download CSV
        logger.info("Downloading knowledge base CSV...")
        response = requests.get(DOWNLOAD_URL, timeout=30)
        response.raise_for_status()
        
        with open("kb.csv", "wb") as f:
            f.write(response.content)
        
        # Load CSV
        df = pd.read_csv("kb.csv")
        logger.info(f"CSV loaded successfully. Shape: {df.shape}")
        
        # Reset documents list
        documents = []
        
        # Check the structure
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Sample data:\n{df.head(3)}")
        
        # Clean and prepare data
        df = df.fillna('Not Specified')
        
        # Create meaningful document chunks for each laptop
        for idx, row in df.iterrows():
            # Create a descriptive string for each laptop
            laptop_info = []
            
            # Add brand and processor info first
            laptop_info.append(f"Brand: {row['Brand']}")
            laptop_info.append(f"Processor: {row['Processor']} {row['Core']}")
            laptop_info.append(f"Processor Generation: {row['Generation']}")
            
            # Add RAM info
            laptop_info.append(f"RAM: {row['Ram']}GB {row['Ram Type']}")
            
            # Add storage info
            storage_parts = []
            if int(row['SSD']) > 0:
                storage_parts.append(f"{row['SSD']}GB SSD")
            if int(row['HDD']) > 0:
                storage_parts.append(f"{row['HDD']}GB HDD")
            
            if storage_parts:
                laptop_info.append(f"Storage: {' + '.join(storage_parts)}")
            else:
                laptop_info.append("Storage: Not Specified")
            
            # Add OS info
            laptop_info.append(f"Operating System: {row['OS']} {row['OS-bit']}")
            
            # Add warranty
            laptop_info.append(f"Warranty: {row['Warranty']}")
            
            # Add price (formatted)
            price = f"₹{int(row['Price']):,}"
            laptop_info.append(f"Price: {price}")
            
            # Add rating info if available
            if int(row['Rating']) > 0:
                laptop_info.append(f"Rating: {row['Rating']}/5")
                if int(row['Number of Ratings']) > 0:
                    laptop_info.append(f"Based on {row['Number of Ratings']} ratings")
            
            # Create final document string
            doc = f"Laptop Specification {idx+1}: " + " | ".join(laptop_info)
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} document entries from {len(df)} rows")
        
        if len(documents) == 0:
            return "❌ Error: No documents were created from the CSV"
        
        # Show sample of created documents
        logger.info("Sample documents created:")
        for i in range(min(3, len(documents))):
            logger.info(f"{i+1}. {documents[i][:150]}...")
        
        # Create embeddings
        logger.info("Creating embeddings...")
        vectors = embed_model.encode(documents, convert_to_numpy=True).astype("float32")
        logger.info(f"Embeddings shape: {vectors.shape}")
        
        # Create FAISS index
        embedding_dim = vectors.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(vectors)
        
        logger.info("Knowledge base built successfully!")
        return f"✅ Knowledge base successfully loaded! Found {len(documents)} laptops in database."
    
    except Exception as e:
        logger.error(f"Error building knowledge base: {str(e)}")
        return f"❌ Error loading knowledge base: {str(e)}"


def ask(query):
    if index is None:
        return "⚠️ Please load the knowledge base first by clicking '📥 Load Knowledge Base' button."
    
    try:
        # Create query embedding
        q_vec = embed_model.encode([query]).astype("float32")
        
        # Search for similar documents (increased from 3 to 5 for better context)
        distances, indices = index.search(q_vec, 5)
        
        logger.info(f"\n=== Query: '{query}' ===")
        logger.info(f"Distances: {distances[0]}")
        logger.info(f"Indices found: {indices[0]}")
        
        # Check if we found relevant results
        valid_indices = [idx for idx in indices[0] if idx < len(documents)]
        
        if len(valid_indices) == 0:
            return "😕 No relevant information found in the knowledge base."
        
        # Use adaptive threshold based on distribution
        mean_distance = np.mean(distances[0])
        std_distance = np.std(distances[0])
        
        logger.info(f"Mean distance: {mean_distance:.3f}, Std: {std_distance:.3f}")
        
        # Get relevant documents (using top 3 for context)
        retrieved_docs = []
        for i in range(min(3, len(valid_indices))):
            idx = valid_indices[i]
            if idx < len(documents):
                retrieved_docs.append(documents[idx])
        
        # If all distances are too high, the query might not be relevant
        if distances[0][0] > 3.0:  # Increased threshold
            return f"""😕 The query doesn't seem to match well with the knowledge base.
            
**Query:** {query}

**Best match distance:** {distances[0][0]:.3f} (higher is less relevant)

Try asking about:
- Laptop specifications (RAM, SSD, HDD, Processor)
- Brand comparisons (ASUS, Lenovo, Dell, HP, etc.)
- Price ranges
- Specific features"""

        # Prepare context
        retrieved = "\n---\n".join(retrieved_docs)
        
        # Enhanced prompt with clear instructions
        prompt = f"""You are a helpful laptop expert assistant. Answer the question based ONLY on the context provided below.

**Context Information (Laptop Specifications):**
{retrieved}

**Question:** {query}

**Instructions:**
1. Answer strictly based on the context above
2. If the exact answer is not in the context, say "I don't have enough information about that in my knowledge base"
3. Be specific about specifications when available
4. Format prices as Indian Rupees (₹)
5. If comparing multiple laptops, present them clearly

**Answer:**"""
        
        # Get response from LLM
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a knowledgeable laptop specification expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = completion.choices[0].message.content.strip()
        
        # Format the response nicely
        response = f"""
### 🤖 Answer
{answer}

---

### 📊 Retrieval Information
• Found {len(valid_indices)} relevant laptop specifications
• Best match distance: {distances[0][0]:.3f} (lower is better)
"""
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"❌ Error processing your query: {str(e)}"


def get_sample_questions():
    """Return sample questions for users"""
    return [
        "Show me laptops under ₹40,000",
        "Compare ASUS and Lenovo laptops with 8GB RAM",
        "What are the best laptops with SSD storage?",
        "Find laptops with Intel Core i5 processor",
        "Show Dell laptops with 1TB HDD",
        "Which laptops have Windows 11?",
        "Find gaming laptops with good ratings",
        "Show laptops with warranty more than 1 year",
        "Compare prices of HP and ASUS laptops",
        "Find laptops with Ryzen processors"
    ]


def search_by_specs(brand=None, min_price=None, max_price=None, min_ram=None, processor_type=None):
    """Direct search function for specific specifications"""
    if df is None:
        return "Please load the knowledge base first."
    
    try:
        filtered_df = df.copy()
        
        # Apply filters
        if brand and brand != "Any":
            filtered_df = filtered_df[filtered_df['Brand'].str.contains(brand, case=False, na=False)]
        
        if min_price:
            filtered_df = filtered_df[filtered_df['Price'] >= int(min_price)]
        
        if max_price:
            filtered_df = filtered_df[filtered_df['Price'] <= int(max_price)]
        
        if min_ram:
            filtered_df = filtered_df[filtered_df['Ram'] >= int(min_ram)]
        
        if processor_type and processor_type != "Any":
            filtered_df = filtered_df[filtered_df['Core'].str.contains(processor_type, case=False, na=False)]
        
        if len(filtered_df) == 0:
            return "No laptops found matching your criteria."
        
        # Format results
        results = []
        for idx, row in filtered_df.head(10).iterrows():
            laptop_desc = f"""
**{row['Brand']} Laptop**
• Processor: {row['Processor']} {row['Core']} ({row['Generation']})
• RAM: {row['Ram']}GB {row['Ram Type']}
• Storage: {row['SSD']}GB SSD + {row['HDD']}GB HDD
• OS: {row['OS']} {row['OS-bit']}
• Warranty: {row['Warranty']}
• Price: ₹{int(row['Price']):,}
• Rating: {row['Rating']}/5 ({row['Number of Ratings']} ratings)
"""
            results.append(laptop_desc)
        
        return f"**Found {len(filtered_df)} laptops:**\n\n" + "\n---\n".join(results)
    
    except Exception as e:
        return f"Error in search: {str(e)}"


############## Enhanced UI with HCI Principles ##############

css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 20px;
}
.main-container {
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}
.header {
    text-align: center;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 2px solid #e0e0e0;
}
.header h1 {
    color: #2c3e50;
    margin-bottom: 10px;
    font-size: 2.5em;
}
.header p {
    color: #7f8c8d;
    font-size: 1.1em;
}
.status-box {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    border-left: 4px solid #4A90E2;
}
#answer-markdown {
    background: white;
    color: #2c3e50;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    margin-top: 20px;
}
button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
}
button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
}
.sample-questions {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
}
.sample-questions h4 {
    color: #2c3e50;
    margin-bottom: 10px;
}
.sample-questions .question-chip {
    display: inline-block;
    background: white;
    padding: 8px 15px;
    margin: 5px;
    border-radius: 20px;
    border: 1px solid #e0e0e0;
    cursor: pointer;
    transition: all 0.2s ease;
}
.sample-questions .question-chip:hover {
    background: #667eea;
    color: white;
    transform: translateY(-1px);
}
.filter-section {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
}
.filter-row {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
    margin-bottom: 15px;
}
.filter-item {
    flex: 1;
    min-width: 200px;
}
"""

with gr.Blocks(css=css, title="Laptop RAG Assistant") as ui:
    with gr.Column(elem_classes="main-container"):
        # Header
        with gr.Column(elem_classes="header"):
            gr.Markdown("# 🖥️ Laptop Specification Assistant")
            gr.Markdown("### AI-Powered RAG System with 300+ Laptop Database")
            gr.Markdown("Ask questions about laptop specifications, prices, brands, and features!")
        
        # Knowledge Base Loading Section
        with gr.Row():
            with gr.Column(scale=2):
                load_btn = gr.Button("📥 Load Knowledge Base (300+ Laptops)", 
                                   elem_classes="primary", size="lg")
            with gr.Column(scale=3):
                status = gr.Textbox(label="System Status", 
                                   value="⚠️ Knowledge base not loaded yet. Click the button to load.",
                                   elem_classes="status-box")
        
        # Sample Questions
        with gr.Column(elem_classes="sample-questions"):
            gr.Markdown("### 💡 Try these sample questions:")
            sample_questions = get_sample_questions()
            question_chips = []
            for q in sample_questions[:5]:  # Show first 5
                chip = gr.Button(q, size="sm", elem_classes="question-chip")
                question_chips.append(chip)
        
        # Filter Section (Optional Advanced Search)
        with gr.Column(elem_classes="filter-section", visible=False) as filter_section:
            gr.Markdown("### 🔍 Advanced Search Filters")
            with gr.Row(elem_classes="filter-row"):
                brand = gr.Dropdown(
                    label="Brand",
                    choices=["Any", "ASUS", "Lenovo", "DELL", "HP", "acer", "MSI", "Avita"],
                    value="Any"
                )
                processor_type = gr.Dropdown(
                    label="Processor Type",
                    choices=["Any", "Core 3", "Core 5", "Core 7", "Celeron", "Pentium", "Ryzen 3", "Ryzen 5", "Ryzen 7"],
                    value="Any"
                )
            with gr.Row(elem_classes="filter-row"):
                min_ram = gr.Slider(label="Minimum RAM (GB)", minimum=4, maximum=32, step=4, value=4)
                min_price = gr.Number(label="Minimum Price (₹)", value=20000)
                max_price = gr.Number(label="Maximum Price (₹)", value=100000)
            filter_btn = gr.Button("🔍 Filter Laptops", elem_classes="primary")
        
        # Main Query Section
        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    label="Ask a Question About Laptops",
                    placeholder="e.g., 'Show me ASUS laptops under ₹50,000 with 8GB RAM'...",
                    lines=2
                )
            with gr.Column(scale=1):
                search_btn = gr.Button("🔍 Get Answer", elem_classes="primary", size="lg")
        
        # Results Section
        answer = gr.Markdown(
            label="Answer",
            value="👈 Enter a question or click a sample question above",
            elem_id="answer-markdown"
        )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        <div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
        <p>🤖 Powered by FAISS + Sentence Transformers + Groq LLM</p>
        <p>📊 Database: 300+ Laptop Specifications | 💡 Ask about brands, prices, specs!</p>
        </div>
        """)
    
    # Event handlers
    def update_query(question):
        return question
    
    # Connect sample questions to query box
    for chip in question_chips:
        chip.click(
            fn=update_query,
            inputs=[chip],
            outputs=query
        )
    
    # Main functionality
    load_btn.click(
        fn=build_knowledge_base,
        outputs=status
    )
    
    search_btn.click(
        fn=ask,
        inputs=query,
        outputs=answer
    )
    
    # Filter functionality (optional)
    filter_btn.click(
        fn=search_by_specs,
        inputs=[brand, min_price, max_price, min_ram, processor_type],
        outputs=answer
    )
    
    # Allow pressing Enter to search
    query.submit(
        fn=ask,
        inputs=query,
        outputs=answer
    )

# Launch the app
if __name__ == "__main__":
    # Print startup message
    print("=" * 70)
    print("🚀 Starting Laptop RAG Assistant")
    print("=" * 70)
    print("\n📊 Features:")
    print("• 300+ Laptop specifications database")
    print("• Natural language query processing")
    print("• Vector similarity search with FAISS")
    print("• LLM-powered responses via Groq")
    print("\n🌐 The app will launch in your browser...")
    print("=" * 70)
    
    try:
        ui.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7861,
            show_error=True,
            debug=True
        )
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("\n💡 If you get a port error, try:")
        print("1. Close other Gradio apps")
        print("2. Change server_port to a different number (e.g., 7861)")
        print("3. Run: ui.launch(share=False) for local only")
