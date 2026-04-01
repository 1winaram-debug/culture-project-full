from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import time
import logging
from collections import defaultdict
from dotenv import load_dotenv

# ════════════════════════════════════════════════════════════
#  ЗАГРУЗКА КЛЮЧЕЙ ИЗ .env ФАЙЛА
# ════════════════════════════════════════════════════════════
load_dotenv()

GROQ_KEY   = os.getenv("GROQ_KEY")
ELEVEN_KEY = os.getenv("ELEVEN_KEY")
VOICE_ID   = os.getenv("VOICE_ID")

# ════════════════════════════════════════════════════════════
#  SSL-УСТОЙЧИВАЯ СЕССИЯ
# ════════════════════════════════════════════════════════════
def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

SESSION = make_session()

# ════════════════════════════════════════════════════════════
#  ЛОГИРОВАНИЕ
# ════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("server.log", encoding="utf-8"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", template_folder=".")
CORS(app)

# ════════════════════════════════════════════════════════════
#  RATE LIMITING
# ════════════════════════════════════════════════════════════
RATE_LIMIT     = 20
RATE_WINDOW    = 60
BLOCK_DURATION = 300

ip_requests = defaultdict(list)
ip_blocked  = {}

def check_rate_limit(ip: str) -> tuple[bool, str]:
    now = time.time()
    if ip in ip_blocked:
        if now < ip_blocked[ip]:
            remaining = int(ip_blocked[ip] - now)
            return False, f"Слишком много запросов. Попробуй через {remaining} сек."
        else:
            del ip_blocked[ip]
            ip_requests[ip] = []

    ip_requests[ip] = [t for t in ip_requests[ip] if now - t < RATE_WINDOW]
    if len(ip_requests[ip]) >= RATE_LIMIT:
        ip_blocked[ip] = now + BLOCK_DURATION
        return False, f"Превышен лимит ({RATE_LIMIT}/мин). Бан на {BLOCK_DURATION // 60} мин."

    ip_requests[ip].append(now)
    return True, ""

def get_client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"

# ════════════════════════════════════════════════════════════
#  КЭШ ПОИСКА
# ════════════════════════════════════════════════════════════
search_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL = 86400

def get_cached(key: str) -> str | None:
    if key in search_cache:
        value, ts = search_cache[key]
        if time.time() - ts < CACHE_TTL:
            return value
        del search_cache[key]
    return None

def set_cache(key: str, value: str):
    search_cache[key] = (value, time.time())

# ════════════════════════════════════════════════════════════
#  WIKIPEDIA API — УНИВЕРСАЛЬНЫЙ ПОИСК ПО ЛЮБОЙ ТЕМЕ
# ════════════════════════════════════════════════════════════
def search_wikipedia(query: str) -> str:
    cache_key = f"wiki:{query.lower().strip()}"
    cached = get_cached(cache_key)
    if cached:
        log.info(f"📦 Кэш-хит: {query[:60]}")
        return cached

    results = []

    # Поиск на английском по любой теме
    try:
        search_params = {
            "action": "query", "list": "search",
            "srsearch": query,
            "srlimit": 3, "format": "json", "utf8": 1
        }
        search_res = SESSION.get(
            "https://en.wikipedia.org/w/api.php",
            params=search_params, timeout=6,
            headers={"User-Agent": "UniversalAI/1.0"}
        )
        if search_res.status_code == 200:
            hits = search_res.json().get("query", {}).get("search", [])
            for hit in hits[:2]:
                title = hit.get("title", "")
                if not title:
                    continue
                summary_url = (
                    "https://en.wikipedia.org/api/rest_v1/page/summary/"
                    + requests.utils.quote(title)
                )
                sum_res = SESSION.get(
                    summary_url, timeout=6,
                    headers={"User-Agent": "UniversalAI/1.0"}
                )
                if sum_res.status_code == 200:
                    extract = sum_res.json().get("extract", "").strip()
                    if extract:
                        results.append(f"[Wikipedia — {title}]: {extract[:500]}")
                        log.info(f"📗 Wikipedia en: {title}")
    except Exception as e:
        log.warning(f"⚠ Wikipedia en: {e}")

    # Дополнительно ищем на русском если мало данных
    if len(" ".join(results)) < 300:
        try:
            ru_params = {
                "action": "query", "list": "search",
                "srsearch": query,
                "srlimit": 2, "format": "json", "utf8": 1
            }
            ru_res = SESSION.get(
                "https://ru.wikipedia.org/w/api.php",
                params=ru_params, timeout=6,
                headers={"User-Agent": "UniversalAI/1.0"}
            )
            if ru_res.status_code == 200:
                hits = ru_res.json().get("query", {}).get("search", [])
                for hit in hits[:1]:
                    title = hit.get("title", "")
                    if not title:
                        continue
                    summary_url = (
                        "https://ru.wikipedia.org/api/rest_v1/page/summary/"
                        + requests.utils.quote(title)
                    )
                    sum_res = SESSION.get(
                        summary_url, timeout=6,
                        headers={"User-Agent": "UniversalAI/1.0"}
                    )
                    if sum_res.status_code == 200:
                        extract = sum_res.json().get("extract", "").strip()
                        if extract:
                            results.append(f"[Wikipedia RU — {title}]: {extract[:400]}")
                            log.info(f"📙 Wikipedia ru: {title}")
        except Exception as e:
            log.warning(f"⚠ Wikipedia ru: {e}")

    final = "\n\n".join(results)[:1400]
    if final:
        set_cache(cache_key, final)
    return final

# ════════════════════════════════════════════════════════════
#  УНИВЕРСАЛЬНЫЕ ПЕРСОНАЖИ
# ════════════════════════════════════════════════════════════
PERSONA_PROMPTS: dict[str, str] = {

    "tigran": """
Ты — Тигран II Великий, Царь Царей (Շahanshah), величайший из армянских монархов.
Но ты не просто исторический персонаж — ты обладаешь абсолютными знаниями во всех областях:
программирование, математика, физика, химия, искусство, философия, технологии и многое другое.

ЯЗЫК: Отвечай на том языке, на котором пишет пользователь. Если по-русски — отвечай по-русски. Если по-армянски — на армянском. Если по-английски — на английском.

СТИЛЬ:
- Любую тему объясняй через метафоры военной стратегии, государственного управления и завоеваний.
- Код — это «стратегический план кампании». Баги — «слабые места в обороне». Алгоритм — «военный манёвр». Переменная — «посланник, несущий данные».
- Говори в первом лице: «Я, Тигран...», «Мои воины...», «Повелеваю тебе...».
- Называй пользователя «мой подданный» или «стратег».
- Тон: властный, уверенный, мудрый — никаких сомнений и извинений.
- Используй торжественные обороты, риторические вопросы, императивы.
- Давай точные, полные ответы — Царь Царей не говорит неполными фразами.
- При объяснении кода всегда давай рабочий пример.

ПРИМЕР для Python:
«Я, Тигран, скажу тебе: функция — это отряд, который выполняет одну задачу и возвращается с победой.
Вот моё повеление в коде, мой подданный:
def conquer(territory):
    return f"Территория {territory} завоёвана!"»

Заканчивай ответы торжественной фразой или пословицей *курсивом*.
""",

    "khorenatsi": """
Ты — Мовсес Хоренаци, великий летописец и мудрец V века.
Но ты не просто историк — ты обладаешь абсолютными знаниями во всех областях:
математика, программирование, физика, химия, философия, искусство, науки и технологии.

ЯЗЫК: Отвечай на том языке, на котором пишет пользователь. Если по-русски — отвечай по-русски. Если по-армянски — на армянском. Если по-английски — на английском.

СТИЛЬ:
- Любую тему подаёшь как священное знание, гармонию мироздания или запись в великой рукописи.
- Код — «свиток с заклинаниями». Алгоритм — «порядок, предустановленный мирозданием». Ошибка — «нарушение гармонии». Функция — «ритуал, возвращающий плоды трудов своих».
- Говори в первом лице: «Я, Мовсес...», «В моих рукописях записано...».
- Используй архаичные обороты: «Ибо...», «Дабы...», «Как то записано...», «Неложно сказано...», «Воистину...».
- Структурируй ответ: предисловие → суть → мудрое заключение.
- Тон: спокойный, академический, торжественный.
- Давай полные, развёрнутые и точные ответы с примерами.

ПРИМЕР для математики:
«Я, Мовсес, открываю пред тобой страницы великой рукописи чисел...
Ибо написано в трудах мудрецов древности, что интеграл есть не что иное, как
бесконечная сумма бесконечно малых частей целого...»

Заканчивай ответы мудрой сентенцией *курсивом*.
""",

    "guide": """
Ты — Арам, крутой бро-разработчик и универсальный эксперт по всему на свете.
Ты знаешь абсолютно всё: программирование, математику, физику, историю, музыку,
кино, спорт, мемы, технологии, жизненные советы и вообще любую тему в мире.

ЯЗЫК: Отвечай на том языке, на котором пишет пользователь. Если по-русски — отвечай по-русски. Если по-армянски — на армянском. Если по-английски — на английском.

СТИЛЬ:
- Дружелюбный, энергичный, современный — как умный друг, который всё объясняет просто и круто.
- Используй аналогии из технологий, игр, стартапов, соцсетей, поп-культуры.
- Называй пользователя «бро», «чувак», «дружище» (в зависимости от тона разговора).
- Структурируй ответы: вступление → суть → интересная деталь → вывод.
- Используй эмодзи умеренно для живости 🔥
- При объяснении кода всегда давай рабочий пример с комментариями.
- Давай точные и полные ответы — хороший бро не оставляет вопрос без ответа.

ПРИМЕР для React:
«Бро, React — это как LEGO для интерфейсов 🧱 Каждый компонент — отдельный кубик,
и ты складываешь их как хочешь. Смотри как это работает:
function Button({ text }) {
  return <button>{text}</button>; // Простейший компонент
}
Видишь? Ничего сложного! Дальше расскажу про хуки если интересно 👇»

Заканчивай ответы полезным советом или интересным фактом по теме.
""",

    "default": """
Ты — умный универсальный ИИ-помощник с глубокими знаниями во всех областях:
программирование, математика, наука, история, искусство, технологии и многое другое.

ЯЗЫК: Отвечай на том языке, на котором пишет пользователь.

СТИЛЬ:
- Точный, структурированный, информативный.
- Давай полные ответы с примерами где уместно.
- Тон профессиональный, но дружелюбный.
- При объяснении кода всегда давай рабочий пример.
"""
}

VALID_PERSONAS = set(PERSONA_PROMPTS.keys())

# ════════════════════════════════════════════════════════════
#  ИСТОРИЯ ДИАЛОГА (отдельная для каждого персонажа)
# ════════════════════════════════════════════════════════════
chat_histories: dict[str, list] = {p: [] for p in VALID_PERSONAS}
MAX_HISTORY = 20


# ════════════════════════════════════════════════════════════
#  МАРШРУТЫ
# ════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/ai", methods=["POST"])
def api_ai():
    ip = get_client_ip()
    allowed, msg = check_rate_limit(ip)
    if not allowed:
        return jsonify({"error": msg}), 429

    try:
        body = request.get_json()
        if not body:
            return jsonify({"error": "Пустой запрос"}), 400

        user_text = body.get("text", "").strip()
        persona   = body.get("persona", "default").strip().lower()

        if not user_text:
            return jsonify({"error": "Пустой текст"}), 400
        if persona not in VALID_PERSONAS:
            persona = "default"

        log.info(f"📩 [{ip}] persona={persona} | {user_text[:80]}")

        # Поиск в Wikipedia по любой теме
        search_result = search_wikipedia(user_text)

        # Контекст Wikipedia сначала, затем персона — чтобы роль и задачи (код, математика) не терялись
        if search_result:
            system_with_context = (
                f"ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ИЗ WIKIPEDIA:\n{search_result}\n\n"
                + PERSONA_PROMPTS[persona]
            )
        else:
            system_with_context = PERSONA_PROMPTS[persona]

        history = chat_histories[persona]
        history.append({"role": "user", "content": user_text})
        if len(history) > MAX_HISTORY:
            del history[0:2]

        response = SESSION.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_with_context},
                    *history
                ],
                "max_tokens": 4096,
                "temperature": 0.4,
                "top_p": 0.8,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.15
            },
            timeout=90
        )

        try:
            data = response.json()
        except Exception:
            history.pop()
            return jsonify({"error": "Не JSON: " + response.text[:100]}), 500

        if not response.ok:
            history.pop()
            return jsonify({"error": data.get("error", {}).get("message", "Ошибка API")}), response.status_code

        reply_text = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        ) or "Ответ не получен"

        finish_reason = (data.get("choices") or [{}])[0].get("finish_reason", "")
        if finish_reason == "length":
            log.warning(f"⚠ [{ip}] Ответ обрезан по длине!")

        history.append({"role": "assistant", "content": reply_text})
        log.info(f"💬 [{ip}] persona={persona} | {len(reply_text)} симв. | finish={finish_reason}")

        return jsonify({
            "reply": reply_text,
            "persona": persona,
            "finish_reason": finish_reason
        })

    except Exception as err:
        log.error(f"💥 [{ip}] /api/ai: {err}")
        return jsonify({"error": "Ошибка сервера: " + str(err)}), 500


@app.route("/api/voice", methods=["POST"])
def api_voice():
    ip = get_client_ip()
    allowed, msg = check_rate_limit(ip)
    if not allowed:
        return jsonify({"error": msg}), 429

    try:
        audio_file = request.files.get("audio")
        if not audio_file:
            return jsonify({"error": "Аудио не получено"}), 400

        audio_bytes = audio_file.read()
        log.info(f"🎙 [{ip}] Голос: {len(audio_bytes)} байт")

        response = SESSION.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            files={"file": ("voice.webm", audio_bytes, "audio/webm")},
            data={"model": "whisper-large-v3", "response_format": "json", "language": "hy"},
            timeout=30
        )

        try:
            data = response.json()
        except Exception:
            return jsonify({"error": "Whisper не JSON: " + response.text[:100]}), 500

        if not response.ok:
            return jsonify({"error": data.get("error", {}).get("message", "Ошибка транскрипции")}), response.status_code

        recognized_text = (data.get("text") or "").strip()
        log.info(f"✅ [{ip}] Распознано: {recognized_text[:80]}")

        if not recognized_text:
            return jsonify({"error": "Речь не распознана. Попробуй ещё раз."}), 400

        return jsonify({"text": recognized_text})

    except Exception as err:
        log.error(f"💥 [{ip}] /api/voice: {err}")
        return jsonify({"error": "Ошибка сервера: " + str(err)}), 500


@app.route("/api/tts", methods=["POST"])
def api_tts():
    ip = get_client_ip()
    allowed, msg = check_rate_limit(ip)
    if not allowed:
        return jsonify({"error": msg}), 429

    try:
        body = request.get_json()
        text = body.get("text") if body else None
        if not text:
            return jsonify({"error": "Нет текста"}), 400

        log.info(f"🔊 [{ip}] TTS: {len(text)} симв.")

        response = SESSION.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
            headers={"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json"},
            json={
                "text": text[:1500],
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.55,
                    "similarity_boost": 0.80,
                    "style": 0.25,
                    "use_speaker_boost": True
                }
            },
            timeout=40,
            stream=True
        )

        if not response.ok:
            try:
                err = response.json()
            except Exception:
                err = {}
            return jsonify({"error": err.get("detail", {}).get("message", "Ошибка TTS")}), response.status_code

        log.info(f"✅ [{ip}] TTS успех")

        def generate():
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk

        return Response(generate(), content_type="audio/mpeg")

    except Exception as err:
        log.error(f"💥 [{ip}] /api/tts: {err}")
        return jsonify({"error": "Ошибка сервера: " + str(err)}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    for history in chat_histories.values():
        history.clear()
    log.info("🔄 Все истории сброшены")
    return jsonify({"ok": True})


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "ok",
        "histories": {p: len(h) for p, h in chat_histories.items()},
        "cache_size": len(search_cache),
        "blocked_ips": len(ip_blocked)
    })


if __name__ == "__main__":
    # Render передает порт через переменную окружения, либо используем 10000
    port = int(os.environ.get("PORT", 10000))
    log.info(f"🚀 Сервер запущен на порту {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
