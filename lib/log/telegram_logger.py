import os

import telegram_send


class TelegramLogger:

    def log_line(message):
        telegram_line = message

        # Set script path
        file_path = os.path.realpath(__file__)
        script_path = os.path.dirname(file_path)

        # Send line to telegram
        telegram_send.send(messages=[telegram_line], parse_mode="html", conf=os.path.join(script_path, "telegram.config"))
