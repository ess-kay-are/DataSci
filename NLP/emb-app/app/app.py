from shiny import App, ui, reactive, render
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import io
from nltk.tokenize import sent_tokenize
import base64

# Define the UI
app_ui = ui.page_fluid(
    ui.panel_title("Text Embedding and Cosine Similarity App"),
    ui.input_file("file", "Choose a DOCX or CSV file", multiple=False, accept=".docx,.csv"),
    ui.input_select("model", "Choose a Sentence Transformer model", choices=[
        'all-MiniLM-L6-v2',
        'BAAI/bge-small-en-v1.5',
        'BAAI/bge-large-en-v1.5',
        'intfloat/e5-small-v2',
        'intfloat/e5-large-v2'
    ]),
    ui.output_text_verbatim("file_info"),
    ui.output_text_verbatim("model_info"),
    ui.output_text_verbatim("status")
)

# Function to ensure the model is downloaded
def get_model(model_name):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}': {e}")

# Function to extract text sentences from a docx file
def get_text_sentences(file_content):
    doc = Document(io.BytesIO(file_content))
    full_text = [para.text for para in doc.paragraphs]
    full_text_string = '\n'.join(full_text)
    sentences = sent_tokenize(full_text_string)
    return sentences

# Function to extract data from a CSV file
def get_csv_sentences(file_content):
    df = pd.read_csv(io.BytesIO(file_content))
    text_data = df.iloc[:, 0].dropna()
    sentences = []
    for text in text_data:
        sentences.extend(sent_tokenize(str(text)))
    return sentences

# Function to determine file type and extract sentences
def extract_sentences(filename, file_content):
    if filename.endswith('.docx'):
        return get_text_sentences(file_content)
    elif filename.endswith('.csv'):
        return get_csv_sentences(file_content)
    else:
        raise ValueError("Unsupported file type. Please upload a .docx or .csv file.")

# Define the server logic
def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.file, input.model)
    def _():
        file = input.file()
        model_name = input.model()

        if file is None:
            output.file_info.set("No file selected.")
            return

        if model_name is None:
            output.model_info.set("No model selected.")
            return

        output.file_info.set(f"File selected: {file['name']}")
        output.model_info.set(f"Model selected: {model_name}")

        try:
            # Load the selected pre-trained SentenceTransformer model
            model = get_model(model_name)
            
            # Decode the file content
            file_content = base64.b64decode(file['content'].split(",")[1])
            
            # Extract sentences from the file
            sentences = extract_sentences(file['name'], file_content)
            
            # Convert text data into embeddings
            embeddings = model.encode(sentences)

            # Compute cosine similarity between the embeddings
            cosine_similarities = cosine_similarity(embeddings)

            # Convert cosine similarities to a DataFrame for better readability
            cosine_similarities_df = pd.DataFrame(cosine_similarities, index=np.arange(len(sentences)), columns=np.arange(len(sentences)))

            # Export cosine similarities to a CSV file
            output_file_path = 'cosine_similarities.csv'
            cosine_similarities_df.to_csv(output_file_path)

            output.status.set(f"Cosine similarities have been saved to {output_file_path}")
        except Exception as e:
            output.status.set(f"Error: {str(e)}")

# Create the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()

