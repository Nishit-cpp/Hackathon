import os
import uuid
import json
import re
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ============================================================
# GEMINI
# ============================================================

from google import genai


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print()
    print("=" * 70)
    print("ERROR: GEMINI_API_KEY was not found.")
    print()
    print("Create a .env file in the same folder as main.py")
    print("and put:")
    print()
    print("GEMINI_API_KEY=YOUR_API_KEY_HERE")
    print("=" * 70)
    print()


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates")
)


# ============================================================
# GEMINI CLIENT
# ============================================================

client = None

if API_KEY:

    try:

        client = genai.Client(
            api_key=API_KEY
        )

        print("Gemini client initialized.")

    except Exception as error:

        print(
            "Could not initialize Gemini:",
            error
        )


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# SESSION STORAGE
# ============================================================

sessions = {}


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = [
    "English",
    "Telugu",
    "Tamil",
    "Hindi",
    "Bengali",
    "Marathi",
    "Gujarati",
    "Kannada",
    "Malayalam",
    "Odia",
    "Punjabi",
    "Urdu"
]


# ============================================================
# QUESTIONS
# ============================================================

QUESTION_COUNT = 7


QUESTION_TOPICS = [

    "background and current education",

    "interests and things the user enjoys",

    "current skills and previous experience",

    "goals and what the user wants to achieve",

    "financial or practical constraints",

    "preferred learning or work environment",

    "specific problem or opportunity the user wants help with"

]


# ============================================================
# BASIC LANGUAGE INSTRUCTIONS
# ============================================================

LANGUAGE_NAMES = {

    "English":
        "English",

    "Telugu":
        "Telugu (తెలుగు)",

    "Tamil":
        "Tamil (தமிழ்)",

    "Hindi":
        "Hindi (हिन्दी)",

    "Bengali":
        "Bengali (বাংলা)",

    "Marathi":
        "Marathi (मराठी)",

    "Gujarati":
        "Gujarati (ગુજરાતી)",

    "Kannada":
        "Kannada (ಕನ್ನಡ)",

    "Malayalam":
        "Malayalam (മലയാളം)",

    "Odia":
        "Odia (ଓଡ଼ିଆ)",

    "Punjabi":
        "Punjabi (ਪੰਜਾਬੀ)",

    "Urdu":
        "Urdu (اردو)"
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are JanSahay AI.

You are a voice-first Indian skill, education and career
guidance assistant.

Your job is to have a natural conversation with a user and
understand their situation.

The user may speak in an Indian regional language.

The user does NOT have to use exact keywords.

You must understand the meaning of natural sentences.

For example, if someone says in Telugu that they like
computers but have never taken a computer course, you should
understand that they are interested in computers and are
probably a beginner.

Do not force the user to answer using predefined keywords.

Be patient.

Do not ask multiple questions at once.

Ask ONE clear question at a time.

Use simple language suitable for ordinary users.

Do not unnecessarily repeat information the user has already
provided.

The conversation should feel like a helpful human assistant.

At the end, provide a practical recommendation.

Recommendations may include:

- education pathways
- skill-development pathways
- government skill programs
- career areas
- training opportunities
- official government resources
- practical next steps

Never invent an official government scheme or a fake URL.

When providing website links, ONLY use official, verified Indian government portals (e.g., https://www.jansamarth.in/, https://pmvishwakarma.gov.in/, https://www.ncs.gov.in/, https://courses.skillindiadigital.gov.in/courses/).

Do not claim that the user is guaranteed a job, admission,
loan, scholarship, benefit or government assistance.

If important information is missing, clearly say what should
be verified.

The final recommendation must be realistic and actionable.
"""


# ============================================================
# CALL GEMINI
# ============================================================

def ask_gemini(prompt):

    if client is None:

        raise RuntimeError(
            "Gemini client is not available. "
            "Check GEMINI_API_KEY."
        )

    response = client.models.generate_content(

        model=MODEL_NAME,

        contents=prompt

    )

    text = getattr(
        response,
        "text",
        None
    )

    if not text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return text.strip()


# ============================================================
# CLEAN JSON
# ============================================================

def extract_json(text):

    text = text.strip()

    # Remove markdown fences.

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # Find first JSON object.

    start = text.find("{")

    end = text.rfind("}")

    if start == -1 or end == -1:

        raise ValueError(
            "No JSON object found."
        )

    json_text = text[
        start:end + 1
    ]

    return json.loads(
        json_text
    )


# ============================================================
# GET NEXT QUESTION
# ============================================================

def generate_next_question(session):

    language = session["language"]

    question_number = (
        session["question_number"]
    )

    answers = session["answers"]

    previous = []

    for item in answers:

        previous.append(
            {
                "question":
                    item["question"],

                "answer":
                    item["answer"]
            }
        )

    previous_json = json.dumps(
        previous,
        ensure_ascii=False,
        indent=2
    )

    topic = QUESTION_TOPICS[
        min(
            question_number - 1,
            len(QUESTION_TOPICS) - 1
        )
    ]

    prompt = f"""
{SYSTEM_PROMPT}

The user's preferred language is:

{LANGUAGE_NAMES.get(language, language)}

This is question number:

{question_number}

There will be approximately
{QUESTION_COUNT} questions.

The current topic should be:

{topic}

Previous conversation:

{previous_json}

Generate ONE short question.

The question must:

1. Be relevant to the user's previous answers.

2. Ask only one thing.

3. Be easy to answer by voice.

4. Be written in the user's selected language.

5. Avoid asking information that the user already clearly
   provided.

6. Sound natural rather than like a government form.

Return ONLY this JSON:

{{
    "question": "question text"
}}
"""

    raw = ask_gemini(prompt)

    data = extract_json(raw)

    question = data.get(
        "question"
    )

    if not question:

        raise ValueError(
            "Gemini did not return a question."
        )

    return question.strip()


# ============================================================
# START SESSION
# ============================================================

@app.route(
    "/api/start",
    methods=["POST"]
)
def start_session():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        language = data.get(
            "language",
            "English"
        )

        name = data.get(
            "name",
            ""
        ).strip()


        if language not in SUPPORTED_LANGUAGES:

            language = "English"


        session_id = str(
            uuid.uuid4()
        )


        session = {

            "session_id":
                session_id,

            "language":
                language,

            "name":
                name,

            "question_number":
                1,

            "answers":
                [],

            "created":
                True

        }


        sessions[
            session_id
        ] = session


        question = (
            generate_next_question(
                session
            )
        )


        return jsonify({

            "success":
                True,

            "session_id":
                session_id,

            "question":
                question,

            "question_number":
                1,

            "total_questions":
                QUESTION_COUNT

        })


    except Exception as error:

        print(
            "START ERROR:",
            repr(error)
        )

        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ============================================================
# PROCESS ANSWER
# ============================================================

@app.route(
    "/api/message",
    methods=["POST"]
)
def process_message():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        session_id = data.get(
            "session_id"
        )

        message = data.get(
            "message",
            ""
        ).strip()


        if not session_id:

            return jsonify({

                "error":
                    "Session ID is missing."

            }), 400


        if session_id not in sessions:

            return jsonify({

                "error":
                    "Session expired or not found."

            }), 404


        if not message:

            return jsonify({

                "error":
                    "No answer was received."

            }), 400


        session = sessions[
            session_id
        ]


        current_question = (
            generate_question_label(
                session
            )
        )


        session["answers"].append({

            "question":
                current_question,

            "answer":
                message

        })


        # ----------------------------------------------------
        # CHECK WHETHER WE SHOULD FINISH
        # ----------------------------------------------------

        if len(
            session["answers"]
        ) >= QUESTION_COUNT:

            result = (
                generate_final_result(
                    session
                )
            )

            return jsonify({

                "success":
                    True,

                "completed":
                    True,

                "result":
                    result

            })


        # ----------------------------------------------------
        # NEXT QUESTION
        # ----------------------------------------------------

        session[
            "question_number"
        ] += 1


        question = (
            generate_next_question(
                session
            )
        )


        return jsonify({

            "success":
                True,

            "completed":
                False,

            "answer":
                message,

            "question":
                question,

            "question_number":
                session[
                    "question_number"
                ],

            "total_questions":
                QUESTION_COUNT

        })


    except Exception as error:

        print(
            "MESSAGE ERROR:",
            repr(error)
        )

        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ============================================================
# QUESTION LABEL
# ============================================================

def generate_question_label(
    session
):

    number = len(
        session["answers"]
    )

    if number < len(
        QUESTION_TOPICS
    ):

        return (
            QUESTION_TOPICS[number]
        )

    return "User response"


# ============================================================
# FINAL RESULT
# ============================================================

def generate_final_result(
    session
):

    language = session[
        "language"
    ]

    answers = session[
        "answers"
    ]


    answers_json = json.dumps(
        answers,
        ensure_ascii=False,
        indent=2
    )


    prompt = f"""
{SYSTEM_PROMPT}

Now complete the user's assessment.

Selected language:

{LANGUAGE_NAMES.get(language, language)}

User name:

{session.get("name", "")}

Conversation data:

{answers_json}

Analyze the conversation carefully.

Do not simply repeat the user's answers.

Identify:

- their major interests
- their current education
- their approximate skill level
- their experience
- their likely career direction
- practical next steps
- useful skill development
- relevant official resources

The recommendation should be useful even if the user is a
beginner.

Return ONLY valid JSON.

Use exactly this structure. Notice the `resources` array at the end, which you MUST populate with 2 to 3 accurate, official Indian Government URLs related to the solution:

{{
    "title": "short recommended pathway",
    "interest": "main interest",
    "skill": "recommended skill to develop",
    "education": "education or learning pathway",
    "experience": "summary of current experience",
    "career": "recommended career area",
    "support": "practical support or next step",
    "explanation": "clear explanation of why this pathway fits",
    "voice_text": "short natural spoken explanation in the selected language",
    "resources": [
        {{"name": "Portal Name (e.g., JanSamarth)", "url": "[https://www.jansamarth.in/](https://www.jansamarth.in/)"}}
    ]
}}

Important:

- Write every value in the user's selected language except
  proper names, URLs, and official program names when appropriate.
- Keep voice_text natural and easy to understand.
- Do not promise guaranteed outcomes.
- NEVER invent a scheme or a fake URL.
"""


    raw = ask_gemini(
        prompt
    )


    result = extract_json(
        raw
    )


    # --------------------------------------------------------
    # SAFE DEFAULTS
    # --------------------------------------------------------

    fields = [

        "title",

        "interest",

        "skill",

        "education",

        "experience",

        "career",

        "support",

        "explanation",

        "voice_text"

    ]

    for field in fields:
        if field not in result:
            result[field] = "—"
            
    # Ensure resources array exists safely
    if "resources" not in result or not isinstance(result["resources"], list):
        result["resources"] = []


    return result


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "online",

        "gemini":
            client is not None,

        "model":
            MODEL_NAME

    })


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "error":
            "Endpoint not found."

    }), 404


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("        JANSAHAY AI")
    print("        Voice-First Assistant")
    print("=" * 70)
    print()

    if client:

        print(
            "Gemini: CONNECTED"
        )

    else:

        print(
            "Gemini: NOT CONNECTED"
        )

    print()
    print(
        "Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)"
    )
    print()
    print(
        "Press CTRL+C to stop."
    )
    print("=" * 70)
    print()


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )