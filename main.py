#Importing Libraries
import pandas as pd
import torch
import os
import math
import numpy as np
from typing import Dict
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, CSVLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, MarianMTModel, MarianTokenizer


# Initialize models and components
@st.cache_resource
def init_models():
    """
    Initialize and cache required models.
    """
    # Embedding model for vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # TinyLlama model for text generation
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Create a text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=128,  # Short responses for speed
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return embeddings, llm

# Translation dictionaries for UI elements
TRANSLATIONS = {
    'en': {
        'title': "Property Recommendation System",
        'select_role': "Select Your Role",
        'buyer': "Buyer",
        'seller': "Seller",
        'tenant': "Tenant",
        'landlord': "Landlord",
        'to_be_implemented': "This feature is coming soon!",
        'preferences': "Enter Your Preferences",
        'name': "Your Name",
        'area': "Preferred Area",
        'furnishing': "Furnishing Status",
        'transaction': "Transaction Type",
        'status': "Property Status",
        'budget': "Budget (in Rupees)",
        'get_recommendations': "Get Recommendations",
        'hello': "Hello",
        'recommended_properties': "Here are your recommended properties:",
        'property_details': "Property Details",
        'match_quality': "Match Quality",
        'enter_name_warning': "Please enter your name to get recommendations",
        'ask_question': "Ask a question about properties:",
        'get_answer': "Get Answer",
        'finding_answer': "Finding answer...",
        'recommendations_tab': "Recommendations",
        'faq_tab': "FAQ"
    },
    'es': {
        'title': "Sistema de Recomendación de Propiedades",
        'select_role': "Seleccione su Rol",
        'buyer': "Comprador",
        'seller': "Vendedor",
        'tenant': "Inquilino",
        'landlord': "Propietario",
        'to_be_implemented': "¡Esta función estará disponible próximamente!",
        'preferences': "Ingrese sus Preferencias",
        'name': "Su Nombre",
        'area': "Área Preferida",
        'furnishing': "Estado del Amueblado",
        'transaction': "Tipo de Transacción",
        'status': "Estado de la Propiedad",
        'budget': "Presupuesto (en Rupias)",
        'get_recommendations': "Obtener Recomendaciones",
        'hello': "¡Hola",
        'recommended_properties': "Aquí están sus propiedades recomendadas:",
        'property_details': "Detalles de la Propiedad",
        'match_quality': "Calidad de Coincidencia",
        'enter_name_warning': "Por favor ingrese su nombre para obtener recomendaciones",
        'ask_question': "Haga una pregunta sobre propiedades:",
        'get_answer': "Obtener Respuesta",
        'finding_answer': "Buscando respuesta...",
        'recommendations_tab': "Recomendaciones",
        'faq_tab': "Preguntas Frecuentes"
    },
    'fr': {
        'title': "Système de Recommandation de Propriétés",
        'select_role': "Sélectionnez votre Rôle",
        'buyer': "Acheteur",
        'seller': "Vendeur",
        'tenant': "Locataire",
        'landlord': "Propriétaire",
        'to_be_implemented': "Cette fonctionnalité sera bientôt disponible!",
        'preferences': "Entrez vos Préférences",
        'name': "Votre Nom",
        'area': "Zone Préférée",
        'furnishing': "État de l'Ameublement",
        'transaction': "Type de Transaction",
        'status': "Statut de la Propriété",
        'budget': "Budget (en Roupies)",
        'get_recommendations': "Obtenir des Recommandations",
        'hello': "Bonjour",
        'recommended_properties': "Voici vos propriétés recommandées :",
        'property_details': "Détails de la Propriété",
        'match_quality': "Qualité de Correspondance",
        'enter_name_warning': "Veuillez entrer votre nom pour obtenir des recommandations",
        'ask_question': "Posez une question sur les propriétés :",
        'get_answer': "Obtenir une Réponse",
        'finding_answer': "Recherche de réponse...",
        'recommendations_tab': "Recommandations",
        'faq_tab': "FAQ"
    }
}

@st.cache_resource
def load_translation_model(source_lang: str, target_lang: str):
    """
    Load translation model and tokenizer for a language pair.
    """
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text between source and target languages.
    """
    if source_lang == target_lang:
        return text
    
    try:
        model, tokenizer = load_translation_model(source_lang, target_lang)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}. Returning original text.")
        return text

def get_ui_text(key: str, lang: str) -> str:
    """
    Retrieve UI text for the specified language.
    """
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, TRANSLATIONS['en'][key])

def detect_language(text: str) -> str:
    """
    Detect the language of input text using simple word matching.
    """
    spanish_words = set(['hola', 'gracias', 'por', 'favor', 'como'])
    french_words = set(['bonjour', 'merci', 'sil', 'vous', 'comment'])
    
    words = set(text.lower().split())
    
    if any(word in spanish_words for word in words):
        return 'es'
    elif any(word in french_words for word in words):
        return 'fr'
    return 'en'

# Create vector stores
@st.cache_resource
def create_vector_stores():
    """
    Create vector stores for properties and FAQs.
    """

    embeddings, _ = init_models()
    
    # Load and process property data
    loader = CSVLoader(
        file_path="data.csv",
        csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['Name', 'Area', 'Furnishing', 'Transaction', 'Status', 'Price']
        }
    )
    property_data = loader.load()
    
    # Load and process FAQ data
    faq_loader = TextLoader('property_faq.txt')
    faq_data = faq_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Smaller chunks for faster processing
        chunk_overlap=50,
        length_function=len,
    )
    
    # Split documents
    property_chunks = text_splitter.split_documents(property_data)
    faq_chunks = text_splitter.split_documents(faq_data)
    
    # Create vector stores
    property_vectorstore = FAISS.from_documents(property_chunks, embeddings)
    faq_vectorstore = FAISS.from_documents(faq_chunks, embeddings)
    
    return property_vectorstore, faq_vectorstore

def verify_dependencies():
    """
    Verify all required packages are installed
    """
    required_packages = {
        'sentence_transformers': 'sentence-transformers',
        'transformers': 'transformers',
        'torch': 'torch',
        'langchain': 'langchain',
        'faiss': 'faiss-cpu',
        'pandas': 'pandas'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        st.error(f"Missing required packages. Please install: pip install {' '.join(missing_packages)}")
        return False
    return True

def render_buyer_interface(lang_code):
    """
    Display the buyer interface with property recommendations
    """
    df = pd.read_csv('data.csv')
    
    st.sidebar.header(get_ui_text('preferences', lang_code))
    
    user_preferences = {
        'Name': st.sidebar.text_input(get_ui_text('name', lang_code)),
        'Area': st.sidebar.selectbox(get_ui_text('area', lang_code), df['Area'].unique()),
        'Furnishing': st.sidebar.selectbox(get_ui_text('furnishing', lang_code), df['Furnishing'].unique()),
        'Transaction': st.sidebar.selectbox(get_ui_text('transaction', lang_code), df['Transaction'].unique()),
        'Status': st.sidebar.selectbox(get_ui_text('status', lang_code), df['Status'].unique()),
        'Price': st.sidebar.number_input(get_ui_text('budget', lang_code), min_value=0, value=10000000, step=1000000)
    }
    
    if st.button(get_ui_text('get_recommendations', lang_code)):
        if user_preferences['Name']:
            greeting = f"{get_ui_text('hello', lang_code)} {user_preferences['Name']}! {get_ui_text('recommended_properties', lang_code)}"
            st.write(greeting)
            
            try:
                recommendations = get_property_recommendations(user_preferences, df)
                formatted_recommendations = format_recommendations(recommendations)
                
                for idx, rec in enumerate(formatted_recommendations, 1):
                    with st.expander(f"{get_ui_text('property_details', lang_code)} {idx} - {rec['property']['Area']} " +
                                   f"({get_ui_text('match_quality', lang_code)}: {rec['match_quality']['Overall Match']})"):
                        
                        st.subheader(get_ui_text('property_details', lang_code))
                        for key, value in rec['property'].items():
                            translated_key = translate_text(key, 'en', lang_code)
                            st.write(f"{translated_key}: {value}")
                        
                        st.subheader(get_ui_text('match_quality', lang_code))
                        cols = st.columns(3)
                        for i, (key, value) in enumerate(rec['match_quality'].items()):
                            translated_key = translate_text(key, 'en', lang_code)
                            cols[i % 3].metric(translated_key, value)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning(get_ui_text('enter_name_warning', lang_code))

def render_faq_section(lang_code, qa_chain):
    """
    Render the FAQ section
    """
    user_question = st.text_input(get_ui_text('ask_question', lang_code))
    if user_question and st.button(get_ui_text('get_answer', lang_code)):
        try:
            with st.spinner(get_ui_text('finding_answer', lang_code)):
                detected_lang = detect_language(user_question)
                if detected_lang != 'en':
                    english_question = translate_text(user_question, detected_lang, 'en')
                else:
                    english_question = user_question
                
                result = qa_chain({"query": english_question})
                translated_answer = translate_text(result["result"], 'en', lang_code)
                st.write(translated_answer)
        except Exception as e:
            st.error(f"Error: {str(e)}")

def calculate_area_similarity(target_area, property_area):
    """
    Calculate similarity score for areas using fuzzy matching
    """
    from difflib import SequenceMatcher
    
    # Clean and normalize area names
    target_area = str(target_area).lower().strip()
    property_area = str(property_area).lower().strip()
    
    # Direct match check
    if target_area == property_area:
        return 1.0
        
    # Fuzzy matching for similar area names
    return SequenceMatcher(None, target_area, property_area).ratio()
def parse_indian_price(price_str):
    """
    Convert Indian price format to float value in rupees
    Handles various formats: cr, crore, lac, L, etc.
    """
    try:
        # Remove ₹ symbol and whitespace, convert to lowercase
        price_str = str(price_str).replace('₹', '').lower().strip()
        
        # Extract the numeric part
        number = float(''.join([c for c in price_str.split()[0] if c.isdigit() or c == '.']))
        
        # Convert based on unit
        if any(unit in price_str for unit in ['cr', 'crore']):
            return number * 10000000  
        elif any(unit in price_str for unit in ['lac', 'lakh', 'l']):
            return number * 100000   
        else:
            return number  # Assume raw rupees
            
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unable to parse price: {price_str} - {str(e)}")

def calculate_price_similarity(target_price, property_price_str, tolerance=0.3):
    """
    Calculate similarity score based on price difference
    """
    try:
        # Parse property price
        property_price = parse_indian_price(property_price_str)
        target_price = float(target_price)
        
        # Calculate percentage difference
        price_ratio = property_price / target_price if target_price > 0 else 0
        
        if price_ratio > (1 + tolerance):
            # Property is more expensive than budget
            return max(0, 1 - (price_ratio - 1) / tolerance)
        elif price_ratio < (1 - tolerance):
            # Property is cheaper than budget
            return max(0, 1 - (1 - price_ratio) / tolerance)
        else:
            # Within tolerance range
            return 1 - abs(1 - price_ratio)
            
    except (ValueError, TypeError) as e:
        print(f"Error calculating price similarity: {e}")
        return 0.0

def get_property_recommendations(user_preferences, properties_df, num_recommendations=5):
    """
    Get property recommendations using weighted scoring with improved differentiation
    """
    # Weights for different criteria
    weights = {
        'price': 0.4,    
        'area': 0.3,     
        'furnishing': 0.15,
        'transaction': 0.1,
        'status': 0.05
    }
    
    # Create a copy of the dataframe to avoid modifying original
    scored_properties = properties_df.copy()
    
    def calculate_property_score(row):
        scores = {}
        
        # Price scoring with more granular differentiation
        try:
            price_score = calculate_price_similarity(
                user_preferences['Price'], 
                row['Price'],
                tolerance=0.2  # Reduced tolerance for more differentiation
            )
        except:
            price_score = 0.0
        
        # Area scoring with exact and partial matches
        area_score = calculate_area_similarity(user_preferences['Area'], row['Area'])
        
        # Exact match scoring for categorical variables with partial matching
        furnishing_score = (1.0 if str(row['Furnishing']).lower() == str(user_preferences['Furnishing']).lower() 
                          else 0.5 if str(user_preferences['Furnishing']).lower() in str(row['Furnishing']).lower() 
                          else 0.0)
        
        transaction_score = (1.0 if str(row['Transaction']).lower() == str(user_preferences['Transaction']).lower() 
                           else 0.5 if str(user_preferences['Transaction']).lower() in str(row['Transaction']).lower() 
                           else 0.0)
        
        status_score = (1.0 if str(row['Status']).lower() == str(user_preferences['Status']).lower() 
                       else 0.5 if str(user_preferences['Status']).lower() in str(row['Status']).lower() 
                       else 0.0)
        
        scores = {
            'price': price_score,
            'area': area_score,
            'furnishing': furnishing_score,
            'transaction': transaction_score,
            'status': status_score
        }
        
        # Calculate weighted score
        total_score = sum(score * weights[criterion] for criterion, score in scores.items())
        
        # Add small random factor to break ties (0.1% maximum impact)
        total_score += np.random.uniform(0, 0.001)
        
        return pd.Series({
            'total_score': total_score,
            'match_scores': scores,
            'parsed_price': parse_indian_price(row['Price']) if isinstance(row['Price'], (str, int, float)) else 0
        })
    
    # Calculate scores for all properties
    score_columns = scored_properties.apply(calculate_property_score, axis=1)
    
    # Add score columns to the dataframe
    scored_properties = pd.concat([scored_properties, score_columns], axis=1)
    
    # Remove duplicates based on all columns except scores
    scored_properties = scored_properties.drop_duplicates(
        subset=['Area', 'Furnishing', 'Transaction', 'Status', 'Price'],
        keep='first'
    )
    
    # Sort by total score and get top recommendations
    recommendations = scored_properties.nlargest(num_recommendations, 'total_score')
    
    return recommendations

def format_recommendations(recommendations):
    """
    Format recommendations with detailed match quality explanation and price comparison
    """
    formatted_recommendations = []
    
    for _, prop in recommendations.iterrows():
        scores = prop['match_scores']
        
        # Calculate price difference percentage
        try:
            target_price = float(prop['parsed_price'])
            price_diff_pct = ((target_price - float(prop['parsed_price'])) / float(prop['parsed_price'])) * 100
            price_comparison = f"({'+ ' if price_diff_pct > 0 else ''}{price_diff_pct:.1f}% vs. budget)"
        except:
            price_comparison = "(unable to compare)"
        
        match_details = {
            'property': {
                'Area': prop['Area'],
                'Status': prop['Status'],
                'Furnishing': prop['Furnishing'],
                'Transaction': prop['Transaction'],
                'Price': f"{prop['Price']} {price_comparison}"
            },
            'match_quality': {
                'Overall Match': f"{prop['total_score']*100:.1f}%",
                'Price Match': f"{scores['price']*100:.1f}%",
                'Area Match': f"{scores['area']*100:.1f}%",
                'Furnishing Match': f"{scores['furnishing']*100:.1f}%",
                'Transaction Match': f"{scores['transaction']*100:.1f}%",
                'Status Match': f"{scores['status']*100:.1f}%"
            }
        }
        
        formatted_recommendations.append(match_details)
    
    return formatted_recommendations

def create_qa_chain(vectorstore, llm):
    # Simplified prompt template for faster processing
    prompt_template = """
    Based on this context: {context}
    
    Question: {question}
    Give a brief answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Configure chain for speed
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),  # Only get most relevant document
        return_source_documents=False,  # Don't return sources for speed
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain


def main():
    st.title("BOT")
    
    # Language selection
    language = st.sidebar.selectbox(
        "Select Language",
        ['English', 'Español', 'Français'],
        key='language_selector'
    )
    
    # Map selected language to language code
    lang_code = {'English': 'en', 'Español': 'es', 'Français': 'fr'}[language]
    
    # Role selection
    role = st.sidebar.selectbox(
        get_ui_text('select_role', lang_code),
        [
            get_ui_text('buyer', lang_code),
            get_ui_text('seller', lang_code),
            get_ui_text('tenant', lang_code),
            get_ui_text('landlord', lang_code)
        ]
    )
    
    if not verify_dependencies():
        return
    
    try:
        property_vectorstore, faq_vectorstore = create_vector_stores()
        _, llm = init_models()
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return
    
    qa_chain = create_qa_chain(faq_vectorstore, llm)
    
    # Create tabs for recommendations and FAQ
    tabs = st.tabs([get_ui_text('recommendations_tab', lang_code), get_ui_text('faq_tab', lang_code)])
    
    # Render content based on selected role
    with tabs[0]:  # Recommendations tab
        if role == get_ui_text('buyer', lang_code):
            render_buyer_interface(lang_code)
        else:
            st.info(get_ui_text('to_be_implemented', lang_code))
    
    with tabs[1]:  # FAQ tab
        render_faq_section(lang_code, qa_chain)

if __name__ == "__main__":
    main()