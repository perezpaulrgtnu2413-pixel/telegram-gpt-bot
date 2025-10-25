import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# 🧩 Команды
# -----------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = (
        f"👋 Привет, {user.first_name}!\n\n"
        "Я бот, работающий на GPT-4o. Отправь мне сообщение, голос или фото, "
        "и я постараюсь помочь 😊"
    )
    keyboard = [
        [InlineKeyboardButton("💬 Новый диалог", callback_data="reset")],
    ]
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.callback_query.answer("Контекст очищен.")
    await update.callback_query.edit_message_text("🧹 Контекст сброшен. Можем начать заново!")


# -----------------------------------------------------
# 💬 Обработка текста
# -----------------------------------------------------

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_history = context.user_data.get("history", [])

    # добавляем новое сообщение в историю
    chat_history.append({"role": "user", "content": user_message})
    context.user_data["history"] = chat_history[-10:]  # храним последние 10 сообщений

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=chat_history
        )
        reply_text = response.output[0].content[0].text
        await update.message.reply_text(reply_text)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
        logger.error(e)


# -----------------------------------------------------
# 🎧 Обработка голосовых сообщений
# -----------------------------------------------------

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file = await update.message.voice.get_file()
        file_path = await file.download_to_drive()
        with open(file_path.name, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio
            )
        os.remove(file_path.name)
        text = transcript.text
        await update.message.reply_text(f"🎙 Распознал: {text}")
        # сразу отправляем в GPT
        context.user_data["last_voice"] = text
        fake_update = Update(update.update_id, message=update.message)
        fake_update.message.text = text
        await handle_text(fake_update, context)
    except Exception as e:
        await update.message.reply_text(f"Ошибка обработки голоса: {e}")


# -----------------------------------------------------
# 🖼 Обработка изображений
# -----------------------------------------------------

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo = await update.message.photo[-1].get_file()
        file_path = await photo.download_to_drive()
        with open(file_path.name, "rb") as img:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "user", "content": [
                        {"type": "input_text", "text": "Опиши это изображение"},
                        {"type": "input_image", "image": img.read()}
                    ]}
                ]
            )
        os.remove(file_path.name)
        text = response.output[0].content[0].text
        await update.message.reply_text(f"🖼 {text}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка обработки изображения: {e}")


# -----------------------------------------------------
# ⚙️ Обработка кнопок
# -----------------------------------------------------

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query.data == "reset":
        await reset(update, context)


# -----------------------------------------------------
# 🚀 Запуск бота
# -----------------------------------------------------

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("🤖 Бот запущен и слушает Telegram...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
