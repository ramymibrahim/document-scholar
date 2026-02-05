import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass


@dataclass
class EmailResult:
    success: bool
    error_message: str | None = None


class EmailService:
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        sender_email: str,
        sender_name: str = "Document Scholar",
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.sender_email = sender_email
        self.sender_name = sender_name
        self.use_tls = use_tls

    def send_email(
        self, to_email: str, to_name: str, subject: str, body: str
    ) -> EmailResult:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.sender_name} <{self.sender_email}>"
            msg["To"] = f"{to_name} <{to_email}>"
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            return EmailResult(success=True)
        except smtplib.SMTPException as e:
            return EmailResult(success=False, error_message=str(e))
        except Exception as e:
            return EmailResult(success=False, error_message=f"Unexpected error: {e}")
