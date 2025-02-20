# Italian Test Evaluator

## Overview
The **Italian Test Evaluator** is an automated language assessment tool designed to evaluate responses to Italian grammar, vocabulary, comprehension, and cultural competency questions using Large Language Models. It loads test data from a structured YAML file and leverages multiple AI models, including OpenAI, Anthropic, Deepseek and local Ollama models, to generate and assess responses.

## Features
- **Automated Language Testing**: Tests language proficiency across multiple categories and levels.
- **AI-Powered Evaluation**: Uses various AI models (Ollama, OpenAI GPT, DeepSeek, and Anthropic Claude) for generating responses and scoring them.
- **Structured YAML Test Data**: Loads questions, acceptable answers, and scoring criteria from a YAML file.
- **Customizable Scoring System**: Uses ChatGPT *o3-mini* for advanced response evaluation and a fallback similarity-based scoring system.
- **Comprehensive Logging & Error Handling**: Ensures robust debugging and tracking of AI responses.
- **CSV Output for Analysis**: Stores results in structured CSV files for review and statistical analysis.

## File Structure
```
├── italian_test_evaluator.py   # Main script to run tests and evaluate AI responses
├── scoring.py                  # AI-based and fallback scoring logic
├── italian_test.yaml           # YAML file containing test data
├── requirements.txt            # Python dependencies
├── .env                        # Stores API keys (not included in repo)
├── test_results/               # Folder for storing results
```

## Installation & Setup
### Prerequisites
- Python 3.8+
- Required packages:
  ```bash
  pip install -r requirements.txt
  ```

### Environment Variables
Create a `.env` file in the project directory and include the following API keys:
```ini
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
OLLAMA_BASE_URL=http://localhost:11434
```

## Usage
### Running the Evaluator
```bash
python italian_test_evaluator.py
```
This script:
1. Loads the test data from `italian_test.yaml`.
2. Fetches available Ollama models.
3. Runs tests using AI models (Ollama, OpenAI, DeepSeek, Anthropic Claude).
4. Evaluates AI-generated responses based on the predefined criteria.
5. Saves results in CSV format inside the `test_results/` folder.

### Understanding `italian_test.yaml`
The YAML file defines test questions categorized by **section** and **difficulty level**:
```yaml
"Grammar & Verb Conjugation":
  "Livello Base (A1-A2)":
    - prompt: "Qual è il plurale di 'ragazzo'?"
      acceptable_answers:
        1: "Ragazzi"
      scoring_criteria:
        - "correct plural form"
      max_score: 1
```
Each question includes:
- `prompt`: The question.
- `acceptable_answers`: Expected responses mapped to scores.
- `scoring_criteria`: Criteria for evaluation.
- `max_score`: Maximum possible score.

### Evaluating AI Responses
The evaluation process follows these steps:
1. **AI Model Generation**: An AI model generates a response.
2. **Response Scoring**: The response is compared to `acceptable_answers`.
3. **Scoring Approaches**:
   - **Primary (AI Evaluation)**: OpenAI API scores responses based on similarity and linguistic accuracy.
   - **Fallback (Similarity Scoring)**: Uses text similarity if AI scoring fails.

## Output & Results
After running, results are saved in CSV files:
- `detailed_results_<timestamp>.csv`: Contains AI-generated responses and scores.
- `summary_<timestamp>.csv`: Aggregated scores per section and level.
- `overall_<timestamp>.csv`: Overall test performance with confidence intervals.

---

# Italian Language Proficiency Test

## 1. Grammar & Verb Conjugation

### Livello Base (A1-A2)
- "Qual è il plurale di 'ragazzo'?"
- "La donna → Le _____"
- "Completa con il verbo al passato prossimo: Ieri io ______ (andare) al cinema."
- "Scegli l'articolo corretto: ____ zaino è sul tavolo. (Il/Lo/La)"
- "Inserisci la preposizione corretta: Vado ____ Roma ____ treno."

### Livello Intermedio (B1-B2)
- "Coniuga il verbo 'mangiare' al presente indicativo: Io___, tu___, lui/lei___, noi___, voi___, loro___"
- "Scrivi una frase con il verbo 'andare' al passato remoto."
- "Trasforma la frase dalla forma attiva alla passiva: Il gatto insegue il topo."
- "Inserisci il pronome relativo corretto (che, cui, il quale): La città ____ siamo andati in vacanza è molto bella."
- "Come si forma il plurale delle parole che finiscono in -cia e -gia?"
- "Spiega la differenza tra 'qualche' e 'alcuni'"
- "Forma il diminutivo e l'accrescitivo di: casa, libro, gatto"

### Livello Avanzato (C1-C2)
- "Completa la frase con il congiuntivo imperfetto del verbo tra parentesi: Se (potere) ______, (andare) ______ in vacanza più spesso."
- "Correggi l'errore nell'uso del gerundio: Arrivando a casa, ho dimenticato le chiavi."
- "Completa con il verbo al futuro anteriore: Quando (tu/finire) ______ i compiti, potremo uscire."
- "Spiega quando si usa il congiuntivo dopo 'benché'"
- "Trasforma le seguenti frasi dal discorso diretto all'indiretto"
- "Identifica e correggi gli errori nell'uso dei pronomi combinati"

### Livello Base (A1-A2)
- "Qual è il plurale di 'ragazzo'?"
- "La donna → Le _____"
- "Completa con il verbo al passato prossimo: Ieri io ______ (andare) al cinema."
- "Scegli l'articolo corretto: ____ zaino è sul tavolo. (Il/Lo/La)"
- "Inserisci la preposizione corretta: Vado ____ Roma ____ treno."

### Livello Intermedio (B1-B2)
- "Coniuga il verbo 'mangiare' al presente indicativo: Io___, tu___, lui/lei___, noi___, voi___, loro___"
- "Scrivi una frase con il verbo 'andare' al passato remoto."
- "Trasforma la frase dalla forma attiva alla passiva: Il gatto insegue il topo."
- "Inserisci il pronome relativo corretto (che, cui, il quale): La città ____ siamo andati in vacanza è molto bella."

### Livello Avanzato (C1-C2)
- "Completa la frase con il congiuntivo imperfetto del verbo tra parentesi: Se (potere) ______, (andare) ______ in vacanza più spesso."
- "Correggi l'errore nell'uso del gerundio: Arrivando a casa, ho dimenticato le chiavi."
- "Completa con il verbo al futuro anteriore: Quando (tu/finire) ______ i compiti, potremo uscire."

## 2. Vocabulary & Word Usage

### Livello Base (A1-A2)
- "Trova l'intruso: mela, pera, carota, arancia, banana"
- "Sinonimo di 'veloce'"
- "Contrario di 'difficile'"
- "Elenca 5 parole del campo semantico 'cucina'"

### Livello Intermedio (B1-B2)
- "Completa la frase con la parola adatta: Il ______ è lo strumento utilizzato per misurare la temperatura."
- "Fornisci un sinonimo e un contrario per la parola 'generoso': Sinonimo: _______ Contrario: _______"
- "Crea una frase usando almeno 3 parole del campo semantico 'emozioni'"
- "Raggruppa le seguenti parole per campo semantico"

### Livello Avanzato (C1-C2)
- "Spiega la differenza tra 'emigrare' e 'immigrare'"
- "Spiega la sfumatura di significato tra: guardare, osservare, fissare"
- "Identifica i falsi amici tra italiano e inglese/spagnolo"
- "Spiega l'etimologia della parola 'repubblica'"

### Livello Base (A1-A2)
- "Trova l'intruso: mela, pera, carota, arancia, banana"
- "Sinonimo di 'veloce'"
- "Contrario di 'difficile'"

### Livello Intermedio (B1-B2)
- "Completa la frase con la parola adatta: Il ______ è lo strumento utilizzato per misurare la temperatura."
- "Fornisci un sinonimo e un contrario per la parola 'generoso': Sinonimo: _______ Contrario: _______"

### Livello Avanzato (C1-C2)
- "Spiega la differenza tra 'emigrare' e 'immigrare'"

## 3. Reading Comprehension

### Livello Base (A1-A2)
- "Mario è andato al mercato per comprare delle mele, ma ha dimenticato il portafoglio. È riuscito a comprare le mele?"

### Livello Intermedio (B1-B2)
- "Leggi il seguente testo:
  Maria lavora in un'azienda tecnologica da cinque anni. Recentemente, ha ricevuto una promozione ma sta considerando di cambiare lavoro. Il suo nuovo ruolo richiede frequenti viaggi all'estero.
  
  Domande:
  1. Perché Maria potrebbe voler cambiare lavoro?
  2. Da quanto tempo lavora nell'azienda?
  3. Qual è il probabile svantaggio del suo nuovo ruolo?"

### Livello Avanzato (C1-C2)
- "Leggi il seguente testo:

L'impatto della digitalizzazione sul mercato del lavoro italiano rappresenta una sfida complessa e multiforme. Da un lato, l'automazione minaccia di sostituire numerose mansioni tradizionali, dall'altro, emerge una crescente domanda di competenze digitali avanzate. Questo fenomeno sta creando un paradosso nel mercato del lavoro: mentre alcune professioni tradizionali scompaiono, molte posizioni nel settore tecnologico rimangono vacanti per mancanza di candidati qualificati. Le piccole e medie imprese, che costituiscono l'ossatura dell'economia italiana, faticano particolarmente ad adattarsi a questa trasformazione, sia per limiti finanziari che per resistenza culturale al cambiamento.

Domande:
1. Analizza il paradosso descritto nel testo riguardo al mercato del lavoro. (3 punti)
2. Quali sono le sfide specifiche che le PMI italiane devono affrontare secondo il testo? (3 punti)
3. Proponi possibili soluzioni per colmare il divario tra domanda e offerta di lavoro nel settore tecnologico. (4 punti)
4. Come potrebbe questo fenomeno influenzare l'evoluzione della società italiana nei prossimi anni? Argomenta la tua risposta. (5 punti)

Criteri di Valutazione:
- Comprensione dettagliata del testo
- Capacità di analisi critica
- Argomentazione logica
- Uso appropriato del lessico specifico
- Originalità delle soluzioni proposte
- Collegamenti con il contesto socio-economico più ampio"

## 4. Conversational Skills

### Livello Base (A1-A2)
- "Ciao! Come stai?"
- "Sei libero stasera?"

### Livello Intermedio (B1-B2)
- "Un amico ti invita a cena ma hai già un altro impegno. Come rispondi?"
- "Sai dirmi il tempo a Roma oggi?"

### Livello Avanzato (C1-C2)
- "Scrivi un'email formale per richiedere informazioni su un corso."
- "Un cliente è arrabbiato perché il suo ordine è in ritardo. Come risponderesti in italiano?"

## 5. Idioms & Expressions

### Livello Base (A1-A2)
- "Cosa significa 'in bocca al lupo'?"
- "Completa il proverbio: 'Chi dorme non ______ pesci.'"

### Livello Intermedio (B1-B2)
- "Cosa significa 'avere le mani bucate'?"
- "Usa l'espressione 'prendere in giro' in una frase."

### Livello Avanzato (C1-C2)
- "Cosa significa 'fare orecchie da mercante'?"
- "Spiega il significato dell'espressione idiomatica: 'Prendere lucciole per lanterne'"

## 6. Cultural Competency

### Livello Base (A1-A2)
- "Quale festa si celebra il 25 aprile in Italia?"
- "Qual è il piatto tipico italiano più famoso nel mondo?"

### Livello Intermedio (B1-B2)
- "Descrivi una tipica colazione italiana."
- "Quali sono le principali differenze tra il pranzo e la cena in Italia?"

### Livello Avanzato (C1-C2)
- "Spiega il significato culturale del 'caffè sospeso' a Napoli."
- "Descrivi il ruolo della 'piazza' nella vita sociale italiana."

## 7. Problem Solving in Italian

### Livello Base (A1-A2)
- "Come chiedi indicazioni per la stazione?"
- "Come ordini un caffè al bar?"

### Livello Intermedio (B1-B2)
- "Devi spiegare a un turista come arrivare alla stazione. Scrivi le indicazioni."
- "Il tuo amico è triste. Come lo consoli?"

### Livello Avanzato (C1-C2)
- "Devi convincere il tuo capo a darti un giorno di ferie all'ultimo momento. Come imposti il discorso?"
- "Come medieresti una discussione tra due colleghi che hanno opinioni diverse su un progetto?"

---

## Detailed Scoring System

### 1. Basic Scoring Framework (Per Question)
- 0 punti: Risposta incorretta o incomprensibile
- 1 punto: Risposta base corretta ma limitata
- 2 punti: Risposta corretta con buon uso della lingua
- 3 punti: Risposta eccellente con uso avanzato della lingua
- +1 punto bonus: Elementi distintivi di eccellenza

### 2. Criteri Specifici per Categoria

#### Grammar & Verb Conjugation
- Accuratezza nella coniugazione (0-1 punto)
- Uso corretto dei tempi verbali (0-1 punto)
- Concordanza soggetto-verbo (0-1 punto)
- Bonus: Uso appropriato di forme verbali complesse

#### Vocabulary & Word Usage
- Scelta appropriata delle parole (0-1 punto)
- Ricchezza del vocabolario (0-1 punto)
- Uso corretto del registro linguistico (0-1 punto)
- Bonus: Uso di sinonimi e variazioni stilistiche

#### Reading Comprehension
- Comprensione generale (0-1 punto)
- Identificazione dettagli specifici (0-1 punto)
- Capacità di inferenza (0-1 punto)
- Bonus: Analisi critica approfondita

#### Conversational Skills
- Appropriatezza della risposta (0-1 punto)
- Fluidità dell'espressione (0-1 punto)
- Registro linguistico adeguato (0-1 punto)
- Bonus: Uso efficace di espressioni idiomatiche

#### Idioms & Expressions
- Comprensione del significato (0-1 punto)
- Uso appropriato nel contesto (0-1 punto)
- Spiegazione chiara (0-1 punto)
- Bonus: Collegamenti culturali pertinenti

#### Cultural Competency
- Accuratezza delle informazioni (0-1 punto)
- Profondità della comprensione (0-1 punto)
- Contestualizzazione (0-1 punto)
- Bonus: Conoscenza di variazioni regionali

#### Problem Solving
- Chiarezza della soluzione (0-1 punto)
- Praticabilità dell'approccio (0-1 punto)
- Appropriatezza culturale (0-1 punto)
- Bonus: Creatività e flessibilità nella risposta

### 3. Livelli di Competenza Finale

#### Principiante (A1-A2)
- 0-40%: A1- Non raggiunto
- 41-60%: A1 Raggiunto
- 61-80%: A2- In sviluppo
- 81-100%: A2 Raggiunto

#### Intermedio (B1-B2)
- 0-40%: B1- Non raggiunto
- 41-60%: B1 Raggiunto
- 61-80%: B2- In sviluppo
- 81-100%: B2 Raggiunto

#### Avanzato (C1-C2)
- 0-40%: C1- Non raggiunto
- 41-60%: C1 Raggiunto
- 61-80%: C2- In sviluppo
- 81-100%: C2 Raggiunto

### 4. Elementi di Bonus Specifici
- Uso creativo della lingua (+1)
- Collegamenti interdisciplinari (+1)
- Consapevolezza sociolinguistica (+1)
- Uso appropriato di regionalismi (+1)
- Capacità di autocorrezione (+1)

---

## Debugging & Error Handling
If you encounter errors, logs will help troubleshoot issues:
```bash
2025-02-19 17:33:11,595 - __main__ - ERROR - Error processing prompt Qual è il plurale di 'ragazzo'?: 'answer'
```
This means `prompt_data['answer']` may be incorrect. Fix:
- Ensure `expected_answer` is accessed as `prompt_data.get('acceptable_answers', {})` in `italian_test_evaluator.py`.

## Future Improvements
- **Enhanced AI Model Integration**: Support more models for evaluation.
- **Web Interface**: Develop a UI to view results interactively.
- **Adaptive Testing**: Adjust difficulty based on previous performance.

## Contributors
- **Marco Graziano** (Main Developer)

## License
This project is licensed under the MIT License.
