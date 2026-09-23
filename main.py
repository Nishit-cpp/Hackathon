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


class GeminiQuotaError(RuntimeError):
    """Raised when Gemini rejects a request because the project quota is exhausted."""
    pass

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

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=prompt

        )

    except Exception as error:

        error_text = repr(error)

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
            or "quota" in error_text.lower()
        ):
            raise GeminiQuotaError(
                "Gemini quota is temporarily exhausted. "
                "JanSahay presentation fallback is available."
            ) from error

        raise

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
# PRESENTATION FALLBACK
# ============================================================

DEMO_QUESTIONS = {
    "English": [
        "What is your current education or work background?",
        "What subjects, activities, or areas are you most interested in?",
        "What skills do you already have, and what have you learned so far?",
        "What is the main goal you want to achieve in the next one or two years?",
        "Do you have any financial or practical limitations we should consider?",
        "What kind of learning or work environment would suit you best?",
        "What specific problem or opportunity would you like JanSahay AI to help you with?"
    ],
    "Telugu": [
        "మీ ప్రస్తుత విద్య లేదా పని నేపథ్యం ఏమిటి?",
        "మీకు ఏ విషయాలు, కార్యకలాపాలు లేదా రంగాలపై ఎక్కువ ఆసక్తి ఉంది?",
        "మీకు ఇప్పటికే ఉన్న నైపుణ్యాలు ఏమిటి? ఇప్పటివరకు మీరు ఏమి నేర్చుకున్నారు?",
        "రాబోయే ఒకటి లేదా రెండు సంవత్సరాల్లో మీరు సాధించాలనుకునే ప్రధాన లక్ష్యం ఏమిటి?",
        "మేము పరిగణించాల్సిన ఆర్థిక లేదా ఇతర పరిమితులు ఏమైనా ఉన్నాయా?",
        "మీకు ఏ విధమైన నేర్చుకునే లేదా పని చేసే వాతావరణం బాగా సరిపోతుంది?",
        "మీకు సహాయం చేయడానికి JanSahay AI పరిష్కరించాల్సిన ప్రత్యేక సమస్య లేదా అవకాశం ఏమిటి?"
    ],
    "Tamil": [
        "உங்கள் தற்போதைய கல்வி அல்லது வேலை பின்னணி என்ன?",
        "எந்த பாடங்கள், செயல்பாடுகள் அல்லது துறைகளில் உங்களுக்கு அதிக ஆர்வம் உள்ளது?",
        "உங்களிடம் ஏற்கனவே உள்ள திறன்கள் என்ன? இதுவரை என்ன கற்றுக்கொண்டீர்கள்?",
        "அடுத்த ஒன்று அல்லது இரண்டு ஆண்டுகளில் நீங்கள் அடைய விரும்பும் முக்கிய இலக்கு என்ன?",
        "நாங்கள் கருத்தில் கொள்ள வேண்டிய நிதி அல்லது நடைமுறை கட்டுப்பாடுகள் ஏதேனும் உள்ளதா?",
        "உங்களுக்கு ஏற்ற கற்றல் அல்லது பணிச்சூழல் எப்படிப்பட்டதாக இருக்க வேண்டும்?",
        "JanSahay AI உங்களுக்கு உதவ வேண்டிய குறிப்பிட்ட பிரச்சனை அல்லது வாய்ப்பு என்ன?"
    ],
    "Hindi": [
        "आपकी वर्तमान शिक्षा या काम की पृष्ठभूमि क्या है?",
        "आपकी किन विषयों, गतिविधियों या क्षेत्रों में सबसे अधिक रुचि है?",
        "आपके पास पहले से कौन-कौन से कौशल हैं और आपने अब तक क्या सीखा है?",
        "अगले एक या दो वर्षों में आप कौन-सा मुख्य लक्ष्य हासिल करना चाहते हैं?",
        "क्या कोई आर्थिक या व्यावहारिक सीमाएँ हैं जिन्हें हमें ध्यान में रखना चाहिए?",
        "आपके लिए किस तरह का सीखने या काम करने का माहौल सबसे उपयुक्त रहेगा?",
        "कौन-सी खास समस्या या अवसर में आप JanSahay AI की मदद चाहते हैं?"
    ],
    "Bengali": [
        "আপনার বর্তমান শিক্ষা বা কাজের পটভূমি কী?",
        "কোন বিষয়, কাজ বা ক্ষেত্রে আপনার সবচেয়ে বেশি আগ্রহ?",
        "আপনার ইতিমধ্যে কী কী দক্ষতা আছে এবং এখন পর্যন্ত কী শিখেছেন?",
        "আগামী এক বা দুই বছরে আপনি কোন প্রধান লক্ষ্য অর্জন করতে চান?",
        "আমাদের বিবেচনা করার মতো কোনো আর্থিক বা বাস্তব সীমাবদ্ধতা আছে কি?",
        "আপনার জন্য কী ধরনের শেখার বা কাজের পরিবেশ সবচেয়ে উপযুক্ত?",
        "কোন নির্দিষ্ট সমস্যা বা সুযোগে আপনি JanSahay AI-এর সাহায্য চান?"
    ],
    "Marathi": [
        "तुमची सध्याची शैक्षणिक किंवा कामाची पार्श्वभूमी काय आहे?",
        "तुम्हाला कोणत्या विषयांमध्ये, उपक्रमांमध्ये किंवा क्षेत्रांमध्ये सर्वाधिक रस आहे?",
        "तुमच्याकडे आधीपासून कोणती कौशल्ये आहेत आणि तुम्ही आतापर्यंत काय शिकलात?",
        "पुढील एक किंवा दोन वर्षांत तुम्हाला कोणते मुख्य उद्दिष्ट साध्य करायचे आहे?",
        "आम्ही विचारात घ्याव्यात अशा आर्थिक किंवा व्यावहारिक अडचणी आहेत का?",
        "तुमच्यासाठी कोणते शिकण्याचे किंवा कामाचे वातावरण योग्य ठरेल?",
        "तुम्हाला मदत करण्यासाठी JanSahay AI ने कोणती विशिष्ट समस्या किंवा संधी सोडवावी?"
    ],
    "Gujarati": [
        "તમારી હાલની શિક્ષણ અથવા કામની પૃષ્ઠભૂમિ શું છે?",
        "તમને કયા વિષયો, પ્રવૃત્તિઓ અથવા ક્ષેત્રોમાં સૌથી વધુ રસ છે?",
        "તમારી પાસે પહેલેથી કઈ કુશળતાઓ છે અને અત્યાર સુધી તમે શું શીખ્યા છો?",
        "આગામી એક કે બે વર્ષમાં તમે કયું મુખ્ય લક્ષ્ય પ્રાપ્ત કરવા માંગો છો?",
        "અમારે ધ્યાનમાં લેવાની કોઈ આર્થિક અથવા વ્યવહારિક મર્યાદાઓ છે?",
        "તમારા માટે કયા પ્રકારનું શીખવાનું અથવા કામનું વાતાવરણ યોગ્ય રહેશે?",
        "કઈ ચોક્કસ સમસ્યા અથવા તકમાં તમે JanSahay AIની મદદ ઇચ્છો છો?"
    ],
    "Kannada": [
        "ನಿಮ್ಮ ಪ್ರಸ್ತುತ ಶಿಕ್ಷಣ ಅಥವಾ ಕೆಲಸದ ಹಿನ್ನೆಲೆ ಏನು?",
        "ಯಾವ ವಿಷಯಗಳು, ಚಟುವಟಿಕೆಗಳು ಅಥವಾ ಕ್ಷೇತ್ರಗಳಲ್ಲಿ ನಿಮಗೆ ಹೆಚ್ಚು ಆಸಕ್ತಿ ಇದೆ?",
        "ನಿಮ್ಮಲ್ಲಿರುವ ಕೌಶಲ್ಯಗಳು ಯಾವುವು ಮತ್ತು ನೀವು ಇದುವರೆಗೆ ಏನು ಕಲಿತಿದ್ದೀರಿ?",
        "ಮುಂದಿನ ಒಂದು ಅಥವಾ ಎರಡು ವರ್ಷಗಳಲ್ಲಿ ನೀವು ಸಾಧಿಸಲು ಬಯಸುವ ಮುಖ್ಯ ಗುರಿ ಏನು?",
        "ನಾವು ಪರಿಗಣಿಸಬೇಕಾದ ಯಾವುದೇ ಆರ್ಥಿಕ ಅಥವಾ ಪ್ರಾಯೋಗಿಕ ಮಿತಿಗಳಿವೆಯೇ?",
        "ನಿಮಗೆ ಯಾವ ರೀತಿಯ ಕಲಿಕೆ ಅಥವಾ ಕೆಲಸದ ವಾತಾವರಣ ಸೂಕ್ತವಾಗಿರುತ್ತದೆ?",
        "ಯಾವ ನಿರ್ದಿಷ್ಟ ಸಮಸ್ಯೆ ಅಥವಾ ಅವಕಾಶದಲ್ಲಿ JanSahay AI ನಿಮ್ಮಗೆ ಸಹಾಯ ಮಾಡಬೇಕು?"
    ],
    "Malayalam": [
        "നിങ്ങളുടെ നിലവിലെ വിദ്യാഭ്യാസ അല്ലെങ്കിൽ ജോലി പശ്ചാത്തലം എന്താണ്?",
        "ഏത് വിഷയങ്ങൾ, പ്രവർത്തനങ്ങൾ അല്ലെങ്കിൽ മേഖലകളിലാണ് നിങ്ങൾക്ക് ഏറ്റവും കൂടുതൽ താൽപര്യം?",
        "നിങ്ങൾക്ക് ഇതിനകം ഉള്ള കഴിവുകൾ എന്തൊക്കെയാണ്, ഇതുവരെ എന്താണ് പഠിച്ചത്?",
        "അടുത്ത ഒന്നോ രണ്ടോ വർഷങ്ങളിൽ നിങ്ങൾ നേടാൻ ആഗ്രഹിക്കുന്ന പ്രധാന ലക്ഷ്യം എന്താണ്?",
        "ഞങ്ങൾ പരിഗണിക്കേണ്ട സാമ്പത്തികമോ പ്രായോഗികമോ ആയ പരിമിതികൾ ഉണ്ടോ?",
        "നിങ്ങൾക്ക് അനുയോജ്യമായ പഠന അല്ലെങ്കിൽ ജോലി അന്തരീക്ഷം എങ്ങനെയായിരിക്കണം?",
        "JanSahay AI നിങ്ങളെ സഹായിക്കേണ്ട പ്രത്യേക പ്രശ്നമോ അവസരമോ എന്താണ്?"
    ],
    "Odia": [
        "ଆପଣଙ୍କର ବର୍ତ୍ତମାନ ଶିକ୍ଷା କିମ୍ବା କାମର ପୃଷ୍ଠଭୂମି କଣ?",
        "କେଉଁ ବିଷୟ, କାର୍ଯ୍ୟକଳାପ କିମ୍ବା କ୍ଷେତ୍ରରେ ଆପଣଙ୍କର ଅଧିକ ଆଗ୍ରହ ଅଛି?",
        "ଆପଣଙ୍କ ପାଖରେ ପୂର୍ବରୁ କେଉଁ ଦକ୍ଷତା ଅଛି ଏବଂ ଏପର୍ଯ୍ୟନ୍ତ କଣ ଶିଖିଛନ୍ତି?",
        "ଆଗାମୀ ଏକ କିମ୍ବା ଦୁଇ ବର୍ଷରେ ଆପଣ କେଉଁ ମୁଖ୍ୟ ଲକ୍ଷ୍ୟ ପାଇବାକୁ ଚାହୁଁଛନ୍ତି?",
        "ଆମେ ବିଚାର କରିବାକୁ ଥିବା କୌଣସି ଆର୍ଥିକ କିମ୍ବା ବ୍ୟବହାରିକ ସୀମାବଦ୍ଧତା ଅଛି କି?",
        "ଆପଣଙ୍କ ପାଇଁ କେଉଁ ପ୍ରକାରର ଶିକ୍ଷା କିମ୍ବା କାର୍ଯ୍ୟ ପରିବେଶ ଉପଯୁକ୍ତ?",
        "କେଉଁ ନିର୍ଦ୍ଦିଷ୍ଟ ସମସ୍ୟା କିମ୍ବା ସୁଯୋଗରେ ଆପଣ JanSahay AIର ସାହାଯ୍ୟ ଚାହୁଁଛନ୍ତି?"
    ],
    "Punjabi": [
        "ਤੁਹਾਡੀ ਮੌਜੂਦਾ ਪੜ੍ਹਾਈ ਜਾਂ ਕੰਮ ਦੀ ਪਿਛੋਕੜ ਕੀ ਹੈ?",
        "ਤੁਹਾਨੂੰ ਕਿਹੜੇ ਵਿਸ਼ਿਆਂ, ਗਤੀਵਿਧੀਆਂ ਜਾਂ ਖੇਤਰਾਂ ਵਿੱਚ ਸਭ ਤੋਂ ਵੱਧ ਦਿਲਚਸਪੀ ਹੈ?",
        "ਤੁਹਾਡੇ ਕੋਲ ਪਹਿਲਾਂ ਤੋਂ ਕਿਹੜੀਆਂ ਹੁਨਰਾਂ ਹਨ ਅਤੇ ਤੁਸੀਂ ਹੁਣ ਤੱਕ ਕੀ ਸਿੱਖਿਆ ਹੈ?",
        "ਅਗਲੇ ਇੱਕ ਜਾਂ ਦੋ ਸਾਲਾਂ ਵਿੱਚ ਤੁਸੀਂ ਕਿਹੜਾ ਮੁੱਖ ਟੀਚਾ ਹਾਸਲ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ?",
        "ਕੀ ਕੋਈ ਵਿੱਤੀ ਜਾਂ ਵਿਹਾਰਕ ਸੀਮਾਵਾਂ ਹਨ ਜਿਨ੍ਹਾਂ ਨੂੰ ਸਾਨੂੰ ਧਿਆਨ ਵਿੱਚ ਰੱਖਣਾ ਚਾਹੀਦਾ ਹੈ?",
        "ਤੁਹਾਡੇ ਲਈ ਕਿਹੋ ਜਿਹਾ ਸਿੱਖਣ ਜਾਂ ਕੰਮ ਕਰਨ ਦਾ ਮਾਹੌਲ ਸਭ ਤੋਂ ਢੁੱਕਵਾਂ ਰਹੇਗਾ?",
        "ਕਿਹੜੀ ਖਾਸ ਸਮੱਸਿਆ ਜਾਂ ਮੌਕੇ ਵਿੱਚ ਤੁਸੀਂ JanSahay AI ਦੀ ਮਦਦ ਚਾਹੁੰਦੇ ਹੋ?"
    ],
    "Urdu": [
        "آپ کی موجودہ تعلیم یا کام کا پس منظر کیا ہے؟",
        "آپ کو کن مضامین، سرگرمیوں یا شعبوں میں سب سے زیادہ دلچسپی ہے؟",
        "آپ کے پاس پہلے سے کون سی مہارتیں ہیں اور آپ نے اب تک کیا سیکھا ہے؟",
        "اگلے ایک یا دو سال میں آپ کون سا اہم مقصد حاصل کرنا چاہتے ہیں؟",
        "کیا کوئی مالی یا عملی پابندیاں ہیں جنہیں ہمیں مدنظر رکھنا چاہیے؟",
        "آپ کے لیے کس طرح کا سیکھنے یا کام کرنے کا ماحول زیادہ موزوں ہوگا؟",
        "آپ کس خاص مسئلے یا موقع میں JanSahay AI کی مدد چاہتے ہیں؟"
    ]
}


DEMO_RESULT_TEXT = {
    "English": {
        "title": "Personalized pathway and next steps",
        "interest": "Skill development and career growth",
        "skill": "Digital literacy, communication and job-relevant foundational skills",
        "education": "Start with a suitable government learning or skill-development pathway and verify eligibility before applying.",
        "experience": "Based on the information shared during the assessment.",
        "career": "A skill-based career pathway matched to the user's interests and background",
        "support": "Use the official portals below to check eligibility, applications, courses and employment opportunities.",
        "explanation": "JanSahay AI has created a practical starting pathway from the user's answers. Official portals are provided so the user can verify current eligibility and take the next step directly.",
        "voice_text": "Based on your answers, I recommend starting with the relevant government learning, career and support portals below. Please check the current eligibility and application instructions on each official website."
    },
    "Telugu": {
        "title": "మీకు అనుకూలమైన మార్గం మరియు తదుపరి చర్యలు",
        "interest": "నైపుణ్యాభివృద్ధి మరియు కెరీర్ అభివృద్ధి",
        "skill": "డిజిటల్ పరిజ్ఞానం, కమ్యూనికేషన్ మరియు ఉద్యోగానికి ఉపయోగపడే ప్రాథమిక నైపుణ్యాలు",
        "education": "తగిన ప్రభుత్వ విద్య లేదా నైపుణ్యాభివృద్ధి మార్గాన్ని ప్రారంభించి, దరఖాస్తు చేసే ముందు అర్హతను తనిఖీ చేయండి.",
        "experience": "ఈ అంచనా సమయంలో మీరు పంచుకున్న సమాచారాన్ని ఆధారంగా చేసుకుని.",
        "career": "మీ ఆసక్తులు మరియు నేపథ్యానికి సరిపోయే నైపుణ్య ఆధారిత కెరీర్ మార్గం",
        "support": "అర్హత, దరఖాస్తులు, కోర్సులు మరియు ఉద్యోగ అవకాశాలను తనిఖీ చేయడానికి క్రింది అధికారిక పోర్టల్స్‌ను ఉపయోగించండి.",
        "explanation": "మీ సమాధానాల ఆధారంగా JanSahay AI ఒక ఆచరణాత్మక ప్రారంభ మార్గాన్ని రూపొందించింది. ప్రస్తుత అర్హతను ధృవీకరించి నేరుగా తదుపరి చర్య తీసుకోవడానికి అధికారిక పోర్టల్స్ ఇవ్వబడ్డాయి.",
        "voice_text": "మీ సమాధానాల ఆధారంగా, క్రింద ఉన్న సంబంధిత ప్రభుత్వ విద్య, కెరీర్ మరియు సహాయ పోర్టల్స్‌తో ప్రారంభించండి. ప్రతి అధికారిక వెబ్‌సైట్‌లో ప్రస్తుత అర్హత మరియు దరఖాస్తు సూచనలను తనిఖీ చేయండి."
    },
    "Tamil": {
        "title": "உங்களுக்கு ஏற்ற பாதை மற்றும் அடுத்தடுத்த நடவடிக்கைகள்",
        "interest": "திறன் மேம்பாடு மற்றும் தொழில் வளர்ச்சி",
        "skill": "டிஜிட்டல் அறிவு, தொடர்புத்திறன் மற்றும் வேலைக்குத் தேவையான அடிப்படை திறன்கள்",
        "education": "பொருத்தமான அரசு கல்வி அல்லது திறன் மேம்பாட்டு பாதையைத் தொடங்கி, விண்ணப்பிக்கும் முன் தகுதியைச் சரிபார்க்கவும்.",
        "experience": "இந்த மதிப்பீட்டின் போது நீங்கள் பகிர்ந்த தகவல்களின் அடிப்படையில்.",
        "career": "உங்கள் ஆர்வம் மற்றும் பின்னணிக்கு ஏற்ற திறன் அடிப்படையிலான தொழில் பாதை",
        "support": "தகுதி, விண்ணப்பங்கள், படிப்புகள் மற்றும் வேலை வாய்ப்புகளைப் பார்க்க கீழே உள்ள அதிகாரப்பூர்வ தளங்களைப் பயன்படுத்தவும்.",
        "explanation": "உங்கள் பதில்களின் அடிப்படையில் JanSahay AI ஒரு நடைமுறை தொடக்கப் பாதையை உருவாக்கியுள்ளது. தற்போதைய தகுதியைச் சரிபார்த்து அடுத்த நடவடிக்கையை நேரடியாக எடுக்க அதிகாரப்பூர்வ தளங்கள் வழங்கப்பட்டுள்ளன.",
        "voice_text": "உங்கள் பதில்களின் அடிப்படையில், கீழே உள்ள தொடர்புடைய அரசு கல்வி, தொழில் மற்றும் உதவி தளங்களில் இருந்து தொடங்குங்கள். ஒவ்வொரு அதிகாரப்பூர்வ இணையதளத்திலும் தற்போதைய தகுதி மற்றும் விண்ணப்ப வழிமுறைகளைச் சரிபார்க்கவும்."
    },
    "Hindi": {
        "title": "आपके लिए उपयुक्त मार्ग और अगले कदम",
        "interest": "कौशल विकास और करियर विकास",
        "skill": "डिजिटल साक्षरता, संचार और रोजगार से जुड़े बुनियादी कौशल",
        "education": "उपयुक्त सरकारी शिक्षा या कौशल-विकास मार्ग से शुरुआत करें और आवेदन से पहले पात्रता की जाँच करें।",
        "experience": "इस आकलन के दौरान आपके द्वारा साझा की गई जानकारी के आधार पर।",
        "career": "आपकी रुचि और पृष्ठभूमि से मेल खाता कौशल-आधारित करियर मार्ग",
        "support": "पात्रता, आवेदन, पाठ्यक्रम और रोजगार के अवसरों की जाँच के लिए नीचे दिए गए आधिकारिक पोर्टल का उपयोग करें।",
        "explanation": "आपके उत्तरों के आधार पर JanSahay AI ने एक व्यावहारिक शुरुआती मार्ग बनाया है। वर्तमान पात्रता की पुष्टि करने और सीधे अगला कदम उठाने के लिए आधिकारिक पोर्टल दिए गए हैं।",
        "voice_text": "आपके उत्तरों के आधार पर, नीचे दिए गए संबंधित सरकारी शिक्षा, करियर और सहायता पोर्टल से शुरुआत करें। हर आधिकारिक वेबसाइट पर वर्तमान पात्रता और आवेदन के निर्देश जरूर जाँचें।"
    },
    "Bengali": {
        "title": "আপনার জন্য উপযুক্ত পথ ও পরবর্তী পদক্ষেপ",
        "interest": "দক্ষতা উন্নয়ন ও ক্যারিয়ার উন্নতি",
        "skill": "ডিজিটাল সাক্ষরতা, যোগাযোগ ও চাকরির জন্য প্রয়োজনীয় মৌলিক দক্ষতা",
        "education": "উপযুক্ত সরকারি শিক্ষা বা দক্ষতা উন্নয়নের পথ দিয়ে শুরু করুন এবং আবেদন করার আগে যোগ্যতা যাচাই করুন।",
        "experience": "এই মূল্যায়নের সময় আপনার দেওয়া তথ্যের ভিত্তিতে।",
        "career": "আপনার আগ্রহ ও পটভূমির সঙ্গে সামঞ্জস্যপূর্ণ দক্ষতাভিত্তিক ক্যারিয়ার পথ",
        "support": "যোগ্যতা, আবেদন, কোর্স এবং কর্মসংস্থানের সুযোগ দেখতে নিচের সরকারি পোর্টালগুলি ব্যবহার করুন।",
        "explanation": "আপনার উত্তরগুলির ভিত্তিতে JanSahay AI একটি ব্যবহারিক প্রাথমিক পথ তৈরি করেছে। বর্তমান যোগ্যতা যাচাই করে সরাসরি পরবর্তী পদক্ষেপ নেওয়ার জন্য সরকারি পোর্টাল দেওয়া হয়েছে।",
        "voice_text": "আপনার উত্তর অনুযায়ী নিচের প্রাসঙ্গিক সরকারি শিক্ষা, ক্যারিয়ার এবং সহায়তা পোর্টাল দিয়ে শুরু করুন। প্রতিটি সরকারি ওয়েবসাইটে বর্তমান যোগ্যতা ও আবেদনের নির্দেশনা যাচাই করুন।"
    },
    "Marathi": {
        "title": "तुमच्यासाठी योग्य मार्ग आणि पुढील पावले",
        "interest": "कौशल्य विकास आणि करिअर विकास",
        "skill": "डिजिटल साक्षरता, संवादकौशल्य आणि रोजगारासाठी आवश्यक मूलभूत कौशल्ये",
        "education": "योग्य सरकारी शिक्षण किंवा कौशल्य-विकास मार्गाने सुरुवात करा आणि अर्ज करण्यापूर्वी पात्रता तपासा.",
        "experience": "या मूल्यांकनादरम्यान तुम्ही दिलेल्या माहितीच्या आधारावर.",
        "career": "तुमच्या आवडी आणि पार्श्वभूमीशी जुळणारा कौशल्य-आधारित करिअर मार्ग",
        "support": "पात्रता, अर्ज, अभ्यासक्रम आणि रोजगाराच्या संधी तपासण्यासाठी खालील अधिकृत पोर्टल वापरा.",
        "explanation": "तुमच्या उत्तरांच्या आधारावर JanSahay AI ने एक व्यावहारिक सुरुवातीचा मार्ग तयार केला आहे. सध्याची पात्रता तपासून पुढील पाऊल थेट घेण्यासाठी अधिकृत पोर्टल दिले आहेत.",
        "voice_text": "तुमच्या उत्तरांच्या आधारावर खालील संबंधित सरकारी शिक्षण, करिअर आणि सहाय्य पोर्टलपासून सुरुवात करा. प्रत्येक अधिकृत वेबसाइटवर सध्याची पात्रता आणि अर्जाच्या सूचना तपासा."
    },
    "Gujarati": {
        "title": "તમારા માટે યોગ્ય માર્ગ અને આગળના પગલાં",
        "interest": "કૌશલ્ય વિકાસ અને કારકિર્દી વિકાસ",
        "skill": "ડિજિટલ સાક્ષરતા, સંચાર અને રોજગાર માટે જરૂરી મૂળભૂત કૌશલ્યો",
        "education": "યોગ્ય સરકારી શિક્ષણ અથવા કૌશલ્ય વિકાસ માર્ગથી શરૂઆત કરો અને અરજી કરતા પહેલાં પાત્રતા તપાસો.",
        "experience": "આ મૂલ્યાંકન દરમિયાન તમે આપેલી માહિતીના આધારે.",
        "career": "તમારી રુચિ અને પૃષ્ઠભૂમિને અનુરૂપ કૌશલ્ય આધારિત કારકિર્દી માર્ગ",
        "support": "પાત્રતા, અરજીઓ, અભ્યાસક્રમો અને રોજગારની તકો તપાસવા માટે નીચેના સત્તાવાર પોર્ટલનો ઉપયોગ કરો.",
        "explanation": "તમારા જવાબોના આધારે JanSahay AI એ એક વ્યવહારુ પ્રારંભિક માર્ગ તૈયાર કર્યો છે. વર્તમાન પાત્રતા ચકાસવા અને આગળનું પગલું સીધું લેવા માટે સત્તાવાર પોર્ટલ આપવામાં આવ્યા છે.",
        "voice_text": "તમારા જવાબોના આધારે નીચેના સંબંધિત સરકારી શિક્ષણ, કારકિર્દી અને સહાય પોર્ટલથી શરૂઆત કરો. દરેક સત્તાવાર વેબસાઇટ પર વર્તમાન પાત્રતા અને અરજીની સૂચનાઓ તપાસો."
    },
    "Kannada": {
        "title": "ನಿಮಗಾಗಿ ಸೂಕ್ತ ಮಾರ್ಗ ಮತ್ತು ಮುಂದಿನ ಹಂತಗಳು",
        "interest": "ಕೌಶಲ್ಯ ಅಭಿವೃದ್ಧಿ ಮತ್ತು ವೃತ್ತಿ ಬೆಳವಣಿಗೆ",
        "skill": "ಡಿಜಿಟಲ್ ಸಾಕ್ಷರತೆ, ಸಂವಹನ ಮತ್ತು ಉದ್ಯೋಗಕ್ಕೆ ಅಗತ್ಯವಾದ ಮೂಲಭೂತ ಕೌಶಲ್ಯಗಳು",
        "education": "ಸೂಕ್ತವಾದ ಸರ್ಕಾರಿ ಶಿಕ್ಷಣ ಅಥವಾ ಕೌಶಲ್ಯ ಅಭಿವೃದ್ಧಿ ಮಾರ್ಗದಿಂದ ಪ್ರಾರಂಭಿಸಿ ಮತ್ತು ಅರ್ಜಿ ಸಲ್ಲಿಸುವ ಮೊದಲು ಅರ್ಹತೆಯನ್ನು ಪರಿಶೀಲಿಸಿ.",
        "experience": "ಈ ಮೌಲ್ಯಮಾಪನದ ಸಂದರ್ಭದಲ್ಲಿ ನೀವು ಹಂಚಿಕೊಂಡ ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ.",
        "career": "ನಿಮ್ಮ ಆಸಕ್ತಿ ಮತ್ತು ಹಿನ್ನೆಲೆಗೆ ಹೊಂದುವ ಕೌಶಲ್ಯ ಆಧಾರಿತ ವೃತ್ತಿ ಮಾರ್ಗ",
        "support": "ಅರ್ಹತೆ, ಅರ್ಜಿಗಳು, ಕೋರ್ಸ್‌ಗಳು ಮತ್ತು ಉದ್ಯೋಗ ಅವಕಾಶಗಳನ್ನು ಪರಿಶೀಲಿಸಲು ಕೆಳಗಿನ ಅಧಿಕೃತ ಪೋರ್ಟಲ್‌ಗಳನ್ನು ಬಳಸಿ.",
        "explanation": "ನಿಮ್ಮ ಉತ್ತರಗಳ ಆಧಾರದ ಮೇಲೆ JanSahay AI ಪ್ರಾಯೋಗಿಕ ಆರಂಭಿಕ ಮಾರ್ಗವನ್ನು ರಚಿಸಿದೆ. ಪ್ರಸ್ತುತ ಅರ್ಹತೆಯನ್ನು ಪರಿಶೀಲಿಸಿ ಮುಂದಿನ ಹಂತವನ್ನು ನೇರವಾಗಿ ತೆಗೆದುಕೊಳ್ಳಲು ಅಧಿಕೃತ ಪೋರ್ಟಲ್‌ಗಳನ್ನು ನೀಡಲಾಗಿದೆ.",
        "voice_text": "ನಿಮ್ಮ ಉತ್ತರಗಳ ಆಧಾರದ ಮೇಲೆ ಕೆಳಗಿನ ಸಂಬಂಧಿತ ಸರ್ಕಾರಿ ಶಿಕ್ಷಣ, ವೃತ್ತಿ ಮತ್ತು ಸಹಾಯ ಪೋರ್ಟಲ್‌ಗಳಿಂದ ಪ್ರಾರಂಭಿಸಿ. ಪ್ರತಿಯೊಂದು ಅಧಿಕೃತ ವೆಬ್‌ಸೈಟ್‌ನಲ್ಲಿ ಪ್ರಸ್ತುತ ಅರ್ಹತೆ ಮತ್ತು ಅರ್ಜಿ ಸೂಚನೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ."
    },
    "Malayalam": {
        "title": "നിങ്ങൾക്ക് അനുയോജ്യമായ വഴിയും അടുത്ത ഘട്ടങ്ങളും",
        "interest": "നൈപുണ്യ വികസനവും തൊഴിൽ വളർച്ചയും",
        "skill": "ഡിജിറ്റൽ സാക്ഷരത, ആശയവിനിമയം, തൊഴിൽ ആവശ്യമായ അടിസ്ഥാന കഴിവുകൾ",
        "education": "അനുയോജ്യമായ സർക്കാർ വിദ്യാഭ്യാസ അല്ലെങ്കിൽ നൈപുണ്യ വികസന മാർഗം ആരംഭിച്ച് അപേക്ഷിക്കുന്നതിന് മുമ്പ് യോഗ്യത പരിശോധിക്കുക.",
        "experience": "ഈ വിലയിരുത്തലിൽ നിങ്ങൾ പങ്കുവെച്ച വിവരങ്ങളുടെ അടിസ്ഥാനത്തിൽ.",
        "career": "നിങ്ങളുടെ താൽപര്യങ്ങൾക്കും പശ്ചാത്തലത്തിനും അനുയോജ്യമായ നൈപുണ്യാധിഷ്ഠിത തൊഴിൽ മാർഗം",
        "support": "യോഗ്യത, അപേക്ഷകൾ, കോഴ്സുകൾ, തൊഴിൽ അവസരങ്ങൾ എന്നിവ പരിശോധിക്കാൻ താഴെയുള്ള ഔദ്യോഗിക പോർട്ടലുകൾ ഉപയോഗിക്കുക.",
        "explanation": "നിങ്ങളുടെ ഉത്തരങ്ങളെ അടിസ്ഥാനമാക്കി JanSahay AI ഒരു പ്രായോഗിക തുടക്ക മാർഗം തയ്യാറാക്കിയിട്ടുണ്ട്. നിലവിലെ യോഗ്യത പരിശോധിച്ച് അടുത്ത നടപടി നേരിട്ട് എടുക്കാൻ ഔദ്യോഗിക പോർട്ടലുകൾ നൽകിയിരിക്കുന്നു.",
        "voice_text": "നിങ്ങളുടെ ഉത്തരങ്ങളുടെ അടിസ്ഥാനത്തിൽ താഴെയുള്ള ബന്ധപ്പെട്ട സർക്കാർ വിദ്യാഭ്യാസ, തൊഴിൽ, സഹായ പോർട്ടലുകളിൽ നിന്ന് ആരംഭിക്കുക. ഓരോ ഔദ്യോഗിക വെബ്സൈറ്റിലും നിലവിലെ യോഗ്യതയും അപേക്ഷാ നിർദ്ദേശങ്ങളും പരിശോധിക്കുക."
    },
    "Odia": {
        "title": "ଆପଣଙ୍କ ପାଇଁ ଉପଯୁକ୍ତ ପଥ ଏବଂ ପରବର୍ତ୍ତୀ ପଦକ୍ଷେପ",
        "interest": "ଦକ୍ଷତା ବିକାଶ ଏବଂ କ୍ୟାରିୟର ବିକାଶ",
        "skill": "ଡିଜିଟାଲ ସାକ୍ଷରତା, ଯୋଗାଯୋଗ ଏବଂ ଚାକିରି ପାଇଁ ଆବଶ୍ୟକ ମୌଳିକ ଦକ୍ଷତା",
        "education": "ଉପଯୁକ୍ତ ସରକାରୀ ଶିକ୍ଷା କିମ୍ବା ଦକ୍ଷତା ବିକାଶ ପଥରୁ ଆରମ୍ଭ କରନ୍ତୁ ଏବଂ ଆବେଦନ ପୂର୍ବରୁ ଯୋଗ୍ୟତା ଯାଞ୍ଚ କରନ୍ତୁ।",
        "experience": "ଏହି ମୂଲ୍ୟାୟନ ସମୟରେ ଆପଣ ଦେଇଥିବା ସୂଚନା ଆଧାରରେ।",
        "career": "ଆପଣଙ୍କ ଆଗ୍ରହ ଏବଂ ପୃଷ୍ଠଭୂମି ସହିତ ମେଳ ଖାଉଥିବା ଦକ୍ଷତା-ଭିତ୍ତିକ କ୍ୟାରିୟର ପଥ",
        "support": "ଯୋଗ୍ୟତା, ଆବେଦନ, ପାଠ୍ୟକ୍ରମ ଏବଂ ନିଯୁକ୍ତି ସୁଯୋଗ ଯାଞ୍ଚ ପାଇଁ ନିମ୍ନଲିଖିତ ସରକାରୀ ପୋର୍ଟାଲ ବ୍ୟବହାର କରନ୍ତୁ।",
        "explanation": "ଆପଣଙ୍କ ଉତ୍ତର ଆଧାରରେ JanSahay AI ଏକ ବ୍ୟବହାରିକ ପ୍ରାରମ୍ଭିକ ପଥ ତିଆରି କରିଛି। ବର୍ତ୍ତମାନ ଯୋଗ୍ୟତା ଯାଞ୍ଚ କରି ପରବର୍ତ୍ତୀ ପଦକ୍ଷେପ ସିଧାସଳଖ ନେବା ପାଇଁ ସରକାରୀ ପୋର୍ଟାଲ ଦିଆଯାଇଛି।",
        "voice_text": "ଆପଣଙ୍କ ଉତ୍ତର ଆଧାରରେ ନିମ୍ନଲିଖିତ ସରକାରୀ ଶିକ୍ଷା, କ୍ୟାରିୟର ଏବଂ ସହାୟତା ପୋର୍ଟାଲରୁ ଆରମ୍ଭ କରନ୍ତୁ। ପ୍ରତ୍ୟେକ ସରକାରୀ ୱେବସାଇଟରେ ବର୍ତ୍ତମାନ ଯୋଗ୍ୟତା ଏବଂ ଆବେଦନ ନିର୍ଦ୍ଦେଶ ଯାଞ୍ଚ କରନ୍ତୁ।"
    },
    "Punjabi": {
        "title": "ਤੁਹਾਡੇ ਲਈ ਢੁੱਕਵਾਂ ਰਾਹ ਅਤੇ ਅਗਲੇ ਕਦਮ",
        "interest": "ਹੁਨਰ ਵਿਕਾਸ ਅਤੇ ਕਰੀਅਰ ਵਿਕਾਸ",
        "skill": "ਡਿਜ਼ੀਟਲ ਸਾਖਰਤਾ, ਸੰਚਾਰ ਅਤੇ ਰੋਜ਼ਗਾਰ ਨਾਲ ਜੁੜੇ ਬੁਨਿਆਦੀ ਹੁਨਰ",
        "education": "ਢੁੱਕਵੇਂ ਸਰਕਾਰੀ ਸਿੱਖਿਆ ਜਾਂ ਹੁਨਰ-ਵਿਕਾਸ ਰਾਹ ਨਾਲ ਸ਼ੁਰੂ ਕਰੋ ਅਤੇ ਅਰਜ਼ੀ ਤੋਂ ਪਹਿਲਾਂ ਯੋਗਤਾ ਦੀ ਜਾਂਚ ਕਰੋ।",
        "experience": "ਇਸ ਮੁਲਾਂਕਣ ਦੌਰਾਨ ਤੁਹਾਡੇ ਵੱਲੋਂ ਸਾਂਝੀ ਕੀਤੀ ਜਾਣਕਾਰੀ ਦੇ ਆਧਾਰ 'ਤੇ।",
        "career": "ਤੁਹਾਡੀ ਦਿਲਚਸਪੀ ਅਤੇ ਪਿਛੋਕੜ ਨਾਲ ਮੇਲ ਖਾਂਦਾ ਹੁਨਰ-ਅਧਾਰਿਤ ਕਰੀਅਰ ਰਾਹ",
        "support": "ਯੋਗਤਾ, ਅਰਜ਼ੀਆਂ, ਕੋਰਸਾਂ ਅਤੇ ਰੋਜ਼ਗਾਰ ਦੇ ਮੌਕੇ ਵੇਖਣ ਲਈ ਹੇਠਾਂ ਦਿੱਤੇ ਸਰਕਾਰੀ ਪੋਰਟਲ ਵਰਤੋ।",
        "explanation": "ਤੁਹਾਡੇ ਜਵਾਬਾਂ ਦੇ ਆਧਾਰ 'ਤੇ JanSahay AI ਨੇ ਇੱਕ ਵਿਹਾਰਕ ਸ਼ੁਰੂਆਤੀ ਰਾਹ ਤਿਆਰ ਕੀਤਾ ਹੈ। ਮੌਜੂਦਾ ਯੋਗਤਾ ਦੀ ਪੁਸ਼ਟੀ ਕਰਨ ਅਤੇ ਅਗਲਾ ਕਦਮ ਸਿੱਧਾ ਲੈਣ ਲਈ ਸਰਕਾਰੀ ਪੋਰਟਲ ਦਿੱਤੇ ਗਏ ਹਨ।",
        "voice_text": "ਤੁਹਾਡੇ ਜਵਾਬਾਂ ਦੇ ਆਧਾਰ 'ਤੇ ਹੇਠਾਂ ਦਿੱਤੇ ਸੰਬੰਧਿਤ ਸਰਕਾਰੀ ਸਿੱਖਿਆ, ਕਰੀਅਰ ਅਤੇ ਸਹਾਇਤਾ ਪੋਰਟਲਾਂ ਤੋਂ ਸ਼ੁਰੂ ਕਰੋ। ਹਰ ਸਰਕਾਰੀ ਵੈੱਬਸਾਈਟ 'ਤੇ ਮੌਜੂਦਾ ਯੋਗਤਾ ਅਤੇ ਅਰਜ਼ੀ ਦੀਆਂ ਹਦਾਇਤਾਂ ਜਾਂਚੋ।"
    },
    "Urdu": {
        "title": "آپ کے لیے موزوں راستہ اور اگلے اقدامات",
        "interest": "مہارتوں کی ترقی اور کیریئر کی ترقی",
        "skill": "ڈیجیٹل خواندگی، رابطے کی مہارت اور روزگار سے متعلق بنیادی مہارتیں",
        "education": "موزوں سرکاری تعلیم یا مہارت کی ترقی کے راستے سے شروع کریں اور درخواست دینے سے پہلے اہلیت کی جانچ کریں۔",
        "experience": "اس جائزے کے دوران آپ کی فراہم کردہ معلومات کی بنیاد پر۔",
        "career": "آپ کی دلچسپی اور پس منظر سے مطابقت رکھنے والا مہارت پر مبنی کیریئر راستہ",
        "support": "اہلیت، درخواستوں، کورسز اور روزگار کے مواقع کی جانچ کے لیے نیچے دیے گئے سرکاری پورٹلز استعمال کریں۔",
        "explanation": "آپ کے جوابات کی بنیاد پر JanSahay AI نے ایک عملی ابتدائی راستہ تیار کیا ہے۔ موجودہ اہلیت کی تصدیق اور اگلا قدم براہ راست اٹھانے کے لیے سرکاری پورٹلز فراہم کیے گئے ہیں۔",
        "voice_text": "آپ کے جوابات کی بنیاد پر نیچے دیے گئے متعلقہ سرکاری تعلیم، کیریئر اور معاونت کے پورٹلز سے شروع کریں۔ ہر سرکاری ویب سائٹ پر موجودہ اہلیت اور درخواست کی ہدایات ضرور چیک کریں۔"
    }
}


def presentation_fallback_question(session):
    language = session.get("language", "English")
    number = max(1, session.get("question_number", 1))
    questions = DEMO_QUESTIONS.get(language, DEMO_QUESTIONS["English"])
    return questions[min(number - 1, len(questions) - 1)]


def presentation_fallback_result(session):
    language = session.get("language", "English")
    base = DEMO_RESULT_TEXT.get(language, DEMO_RESULT_TEXT["English"]).copy()

    answers = session.get("answers", [])
    joined = " ".join(str(item.get("answer", "")) for item in answers).lower()

    # Keep the original lightweight personalization logic, but localize
    # the displayed recommendation when the selected language is regional.
    if any(word in joined for word in [
        "computer", "coding", "programming", "software", "ai", "machine learning", "data"
    ]):
        localized_interest = {
            "Telugu": "టెక్నాలజీ, కంప్యూటింగ్ మరియు డిజిటల్ నైపుణ్యాలు",
            "Tamil": "தொழில்நுட்பம், கணினி மற்றும் டிஜிட்டல் திறன்கள்",
            "Hindi": "तकनीक, कंप्यूटिंग और डिजिटल कौशल",
            "Bengali": "প্রযুক্তি, কম্পিউটিং ও ডিজিটাল দক্ষতা",
            "Marathi": "तंत्रज्ञान, संगणक आणि डिजिटल कौशल्ये",
            "Gujarati": "ટેક્નોલોજી, કમ્પ્યુટિંગ અને ડિજિટલ કૌશલ્યો",
            "Kannada": "ತಂತ್ರಜ್ಞಾನ, ಕಂಪ್ಯೂಟಿಂಗ್ ಮತ್ತು ಡಿಜಿಟಲ್ ಕೌಶಲ್ಯಗಳು",
            "Malayalam": "സാങ്കേതികവിദ്യ, കമ്പ്യൂട്ടിംഗ്, ഡിജിറ്റൽ കഴിവുകൾ",
            "Odia": "ପ୍ରଯୁକ୍ତି, କମ୍ପ୍ୟୁଟିଂ ଏବଂ ଡିଜିଟାଲ ଦକ୍ଷତା",
            "Punjabi": "ਤਕਨਾਲੋਜੀ, ਕੰਪਿਊਟਿੰਗ ਅਤੇ ਡਿਜ਼ੀਟਲ ਹੁਨਰ",
            "Urdu": "ٹیکنالوجی، کمپیوٹنگ اور ڈیجیٹل مہارتیں"
        }
        localized_career = {
            "Telugu": "సాఫ్ట్‌వేర్, AI/ML, డేటా లేదా ఇతర టెక్నాలజీ కెరీర్లు",
            "Tamil": "மென்பொருள், AI/ML, தரவு அல்லது பிற தொழில்நுட்பத் தொழில்கள்",
            "Hindi": "सॉफ्टवेयर, AI/ML, डेटा या अन्य तकनीकी करियर",
            "Bengali": "সফটওয়্যার, AI/ML, ডেটা বা অন্যান্য প্রযুক্তি ক্যারিয়ার",
            "Marathi": "सॉफ्टवेअर, AI/ML, डेटा किंवा इतर तंत्रज्ञान क्षेत्रातील करिअर",
            "Gujarati": "સોફ્ટવેર, AI/ML, ડેટા અથવા અન્ય ટેક્નોલોજી કારકિર્દી",
            "Kannada": "ಸಾಫ್ಟ್‌ವೇರ್, AI/ML, ಡೇಟಾ ಅಥವಾ ಇತರ ತಂತ್ರಜ್ಞಾನ ವೃತ್ತಿಗಳು",
            "Malayalam": "സോഫ്റ്റ്‌വെയർ, AI/ML, ഡാറ്റ അല്ലെങ്കിൽ മറ്റ് സാങ്കേതിക തൊഴിൽ മേഖലകൾ",
            "Odia": "ସଫ୍ଟୱେର, AI/ML, ଡାଟା କିମ୍ବା ଅନ୍ୟାନ୍ୟ ପ୍ରଯୁକ୍ତି କ୍ୟାରିୟର",
            "Punjabi": "ਸਾਫਟਵੇਅਰ, AI/ML, ਡਾਟਾ ਜਾਂ ਹੋਰ ਤਕਨਾਲੋਜੀ ਕਰੀਅਰ",
            "Urdu": "سافٹ ویئر، AI/ML، ڈیٹا یا دیگر ٹیکنالوجی کیریئر"
        }
        localized_skill = {
            "Telugu": "ప్రోగ్రామింగ్ ప్రాథమికాలు, సమస్య పరిష్కారం మరియు డిజిటల్ నైపుణ్యాలు",
            "Tamil": "நிரலாக்க அடிப்படைகள், சிக்கல் தீர்வு மற்றும் டிஜிட்டல் திறன்கள்",
            "Hindi": "प्रोग्रामिंग की बुनियाद, समस्या समाधान और डिजिटल कौशल",
            "Bengali": "প্রোগ্রামিংয়ের মৌলিক বিষয়, সমস্যা সমাধান ও ডিজিটাল দক্ষতা",
            "Marathi": "प्रोग्रामिंगची मूलतत्त्वे, समस्या सोडवणे आणि डिजिटल कौशल्ये",
            "Gujarati": "પ્રોગ્રામિંગના મૂળભૂત તત્વો, સમસ્યા ઉકેલ અને ડિજિટલ કૌશલ્યો",
            "Kannada": "ಪ್ರೋಗ್ರಾಮಿಂಗ್ ಮೂಲಭೂತಗಳು, ಸಮಸ್ಯೆ ಪರಿಹಾರ ಮತ್ತು ಡಿಜಿಟಲ್ ಕೌಶಲ್ಯಗಳು",
            "Malayalam": "പ്രോഗ്രാമിംഗ് അടിസ്ഥാനങ്ങൾ, പ്രശ്നപരിഹാരം, ഡിജിറ്റൽ കഴിവുകൾ",
            "Odia": "ପ୍ରୋଗ୍ରାମିଂ ମୌଳିକତା, ସମସ୍ୟା ସମାଧାନ ଏବଂ ଡିଜିଟାଲ ଦକ୍ଷତା",
            "Punjabi": "ਪ੍ਰੋਗਰਾਮਿੰਗ ਦੀਆਂ ਬੁਨਿਆਦਾਂ, ਸਮੱਸਿਆ ਹੱਲ ਅਤੇ ਡਿਜ਼ੀਟਲ ਹੁਨਰ",
            "Urdu": "پروگرامنگ کی بنیادی باتیں، مسئلہ حل کرنے اور ڈیجیٹل مہارتیں"
        }
        if language != "English":
            base["interest"] = localized_interest.get(language, base["interest"])
            base["career"] = localized_career.get(language, base["career"])
            base["skill"] = localized_skill.get(language, base["skill"])
        else:
            base["interest"] = "Technology, computing and digital skills"
            base["career"] = "Software, AI/ML, data or other technology careers"
            base["skill"] = "Programming fundamentals, problem solving and digital skills"

    base["resources"] = [
        {
            "name": "myScheme — Government Scheme Discovery",
            "url": "https://www.myscheme.gov.in/",
            "purpose": "Find government schemes that may match your profile and needs.",
            "how_to_use": "1) Open the portal. 2) Explore schemes using the available filters. 3) Read eligibility and required documents. 4) Follow the official application instructions if eligible."
        },
        {
            "name": "National Career Service (NCS)",
            "url": "https://www.ncs.gov.in/",
            "purpose": "Explore government-supported career services, job opportunities and career guidance.",
            "how_to_use": "1) Open NCS. 2) Register or sign in if required. 3) Complete your profile. 4) Search relevant jobs or career services and follow the official instructions."
        },
        {
            "name": "Skill India Digital",
            "url": "https://www.skillindiadigital.gov.in/",
            "purpose": "Explore government-supported skill development and learning opportunities.",
            "how_to_use": "1) Open Skill India Digital. 2) Search for a relevant course. 3) Check eligibility and course details. 4) Enrol through the official portal when available."
        },
        {
            "name": "JanSamarth — Government Credit and Loan Schemes",
            "url": "https://www.jansamarth.in/",
            "purpose": "Check eligible government-linked credit and loan schemes.",
            "how_to_use": "1) Open JanSamarth. 2) Select the relevant loan or credit category. 3) Enter the requested details. 4) Review eligibility and continue only through the official application process."
        },
        {
            "name": "Apprenticeship India",
            "url": "https://www.apprenticeshipindia.gov.in/",
            "purpose": "Explore apprenticeship opportunities and official apprenticeship services.",
            "how_to_use": "1) Open Apprenticeship India. 2) Create or access your candidate profile. 3) Search opportunities matching your qualification and interests. 4) Check the official requirements before applying."
        }
    ]

    # Resource titles/purpose/how-to-use are kept accurate and actionable;
    # the selected-language summary above is what is spoken/displayed.
    return base


# ============================================================
# GET NEXT QUESTION
# ============================================================

def generate_next_question(session):

    if session.get("demo_mode"):
        return presentation_fallback_question(session)

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

    try:
        raw = ask_gemini(prompt)
    except GeminiQuotaError:
        session["demo_mode"] = True
        return presentation_fallback_question(session)

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
                True,

            "demo_mode":
                False

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

    if session.get("demo_mode"):
        return presentation_fallback_result(session)

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

Use exactly this structure. The `resources` array is REQUIRED and is the user's practical action list. Populate it with 4 to 6 accurate, official Indian Government resources that are specifically relevant to the user's situation. Do NOT give generic links when a more precise official portal is known. For example: use PM-Vidyalaxmi for education loans, myScheme for government schemes, NCS for jobs/career services, Skill India Digital for government courses, Apprenticeship India for apprenticeships, and the exact official scheme portal when the scheme is confidently known.

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
        {{
            "name": "Official portal or scheme name",
            "url": "https://official-government-domain/...",
            "purpose": "Exactly what the user can get or do on this website",
            "how_to_use": "Simple step-by-step instructions for this specific user: 1) ... 2) ... 3) ..."
        }}
    ]
}}

Resource rules:
- Every resource MUST be an official Indian Government portal or an official government-program portal.
- Prefer a direct application/course/search/status page when you are confident it is correct; otherwise use the official portal homepage rather than guessing a deep URL.
- NEVER invent a scheme, URL, application page, eligibility rule, benefit, or deadline.
- Do not use private coaching sites, unofficial blogs, referral sites, or search-result URLs.
- Match each resource to the user's actual need. If the user needs a loan, include an official loan application portal. If they need a government scheme, include the official scheme discovery/application portal. If they need courses, include an official government learning portal. If they need work, include an official employment/apprenticeship portal.
- `purpose` must explain why this exact link is useful.
- `how_to_use` must tell the user what to click, what information to enter, what to check, and what to do next, without claiming guaranteed approval.

Important:

- Write every value in the user's selected language except proper names, URLs, and official program names when appropriate.
- Keep voice_text natural and easy to understand.
- Do not promise guaranteed outcomes.
- NEVER invent a scheme or a fake URL.
"""


    try:
        raw = ask_gemini(
            prompt
        )
    except GeminiQuotaError:
        session["demo_mode"] = True
        return presentation_fallback_result(session)


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
            
    # Ensure resources array exists safely and keep only usable official URLs.
    if "resources" not in result or not isinstance(result["resources"], list):
        result["resources"] = []

    cleaned_resources = []
    allowed_domains = (
        ".gov.in",
        ".nic.in",
        "pmvidyalaxmi.co.in",
        ".gov.in/",
    )

    for resource in result["resources"]:
        if not isinstance(resource, dict):
            continue

        url = str(resource.get("url", "")).strip()
        if not url.startswith("https://"):
            continue

        # Accept only official government domains; the model still decides
        # which precise official resource best matches the user's situation.
        host = re.sub(r"^https://", "", url, flags=re.IGNORECASE).split("/", 1)[0].lower()
        if not (host.endswith(".gov.in") or host.endswith(".nic.in") or host == "pmvidyalaxmi.co.in"):
            continue

        cleaned_resources.append({
            "name": str(resource.get("name", "Official Government Resource")).strip(),
            "url": url,
            "purpose": str(resource.get("purpose", "Official website relevant to this recommendation.")).strip(),
            "how_to_use": str(resource.get("how_to_use", "Open the website, check eligibility and follow the official application instructions.")).strip()
        })

    result["resources"] = cleaned_resources[:6]

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
            MODEL_NAME,

        "presentation_fallback":
            True

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