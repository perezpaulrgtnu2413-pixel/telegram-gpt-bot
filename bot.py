import os
import logging
import tempfile
import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

# -----------------------------------------------------
# 🔧 Настройки
# -----------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# ⚙️ Команды
# -----------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    context.user_data["voice_enabled"] = True
    text = (
        f"👋 Привет, {user.first_name}!\n\n"
        "Я — медицинский ассистент на GPT-4o.\n"
        "Отправь 📷 изображение (ЭКГ, анализ, рентген), 🎤 голосовое сообщение или 💬 текст — я помогу с анализом.\n\n"
        "🗣 Озвучка ответов: включена. Команды:\n"
        "/voice_off — отключить звук\n"
        "/voice_on — включить звук"
    )
    keyboard = [[InlineKeyboardButton("🔄 Сбросить диалог", callback_data="reset")]]
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.callback_query.answer("Контекст очищен ✅")
    await update.callback_query.edit_message_text("Диалог сброшен. Начнём заново!")


async def voice_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["voice_enabled"] = True
    await update.message.reply_text("🔊 Озвучка включена — теперь я буду отвечать голосом!")


async def voice_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["voice_enabled"] = False
    await update.message.reply_text("🔇 Озвучка отключена.")


# -----------------------------------------------------
# 💬 Обработка текста
# -----------------------------------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    history = context.user_data.get("history", [])
    history.append({"role": "user", "content": user_message})
    context.user_data["history"] = history[-10:]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history
        )
        reply_text = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": reply_text})
        context.user_data["history"] = history[-10:]

        # Отправляем текстовый ответ
        await update.message.reply_text(reply_text)

        # Если включена озвучка — создаём и отправляем аудио
        if context.user_data.get("voice_enabled", True):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                audio_path = temp_audio.name

            # 🎤 Генерация речи через OpenAI
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=reply_text
            ) as stream:
                stream.stream_to_file(audio_path)

            await update.message.reply_voice(voice=InputFile(audio_path))
            os.remove(audio_path)

    except Exception as e:
        logger.error(e)
        await update.message.reply_text(f"⚠️ Ошибка: {e}")


# -----------------------------------------------------
# 🎧 Обработка голосовых сообщений (Whisper)
# -----------------------------------------------------
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file = await update.message.voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            file_path = temp_file.name
            await file.download_to_drive(custom_path=file_path)

        with open(file_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )

        os.remove(file_path)
        text = transcript.text.strip()
        await update.message.reply_text(f"🎧 Расшифровал: {text}")

        fake_update = Update(update.update_id, message=update.message)
        fake_update.message.text = text
        await handle_text(fake_update, context)

    except Exception as e:
        logger.error(e)
        await update.message.reply_text(f"Ошибка при обработке голоса: {e}")


# -----------------------------------------------------
# 🖼 Обработка изображений
# -----------------------------------------------------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo = await update.message.photo[-1].get_file()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            file_path = temp_file.name
            await photo.download_to_drive(custom_path=file_path)

        image = cv2.imread(file_path)
        if image is not None:
            image = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
            image = cv2.GaussianBlur(image, (3, 3), 0)
            cv2.imwrite(file_path, image)

        with open(file_path, "rb") as img:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Проанализируй это медицинское изображение (ЭКГ, анализ или рентген)."},
                            {"type": "image", "image": img.read()}
                        ]
                    }
                ]
            )

        os.remove(file_path)
        reply_text = response.choices[0].message.content
        await update.message.reply_text(f"📊 {reply_text}")

        # 🎧 Озвучка результата
        if context.user_data.get("voice_enabled", True):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                audio_path = temp_audio.name
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="verse",
                input=reply_text
            ) as stream:
                stream.stream_to_file(audio_path)

            await update.message.reply_voice(voice=InputFile(audio_path))
            os.remove(audio_path)

    except Exception as e:
        logger.error(e)
        await update.message.reply_text(f"Ошибка при анализе изображения: {e}")


# -----------------------------------------------------
# 🚀 Запуск бота
# -----------------------------------------------------
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("voice_on", voice_on))
    app.add_handler(CommandHandler("voice_off", voice_off))
    app.add_handler(CallbackQueryHandler(reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("🤖 Медицинский GPT-бот запущен с озвучкой.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
