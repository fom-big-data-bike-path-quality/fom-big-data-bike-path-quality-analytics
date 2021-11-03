import os
from pathlib import Path

import telegram_send


class TelegramLogger:

    def log_line(self, logger, message, images=None):
        # Set script path
        file_path = os.path.realpath(__file__)
        script_path = os.path.dirname(file_path)

        # Check for config file
        if not Path(os.path.join(script_path, "telegram.config")).exists():
            logger.log_line("✗️ Telegram config not found " + os.path.join(script_path, "telegram.config"))
            return

        # Send line to telegram
        telegram_send.send(
            messages=[message],
            images=images,
            parse_mode="html",
            conf=os.path.join(script_path, "telegram.config")
        )
