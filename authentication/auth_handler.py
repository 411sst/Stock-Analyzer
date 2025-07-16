import streamlit as st
import psycopg2
import bcrypt
import validators
import re
from datetime import datetime

class AuthHandler:
    def __init__(self):
        # Get the connection string from Streamlit secrets
        self.db_url = st.secrets["database"]["connection_string"]
        self._initialize_db()

    def _get_connection(self):
        """Establish a connection to the PostgreSQL database."""
        return psycopg2.connect(self.db_url)

    def _initialize_db(self):
        """Initialize database tables"""
        # Connect to the PostgreSQL database
        conn = self._get_connection()
        cursor = conn.cursor()

        # Users table (PostgreSQL syntax)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP WITH TIME ZONE,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)

        # User portfolios table (PostgreSQL syntax)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_portfolios (
                id SERIAL PRIMARY KEY,
                user_id INTEGER,
                symbol TEXT,
                quantity REAL,
                buy_price REAL,
                buy_date DATE,
                notes TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # User watchlists table (PostgreSQL syntax)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_watchlists (
                id SERIAL PRIMARY KEY,
                user_id INTEGER,
                symbol TEXT,
                added_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                alert_price REAL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # User preferences table (PostgreSQL syntax)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY,
                theme TEXT DEFAULT 'dark',
                default_mode TEXT DEFAULT 'Beginner',
                email_notifications BOOLEAN DEFAULT TRUE,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()

    def register_user(self, username, email, password):
        """Register new user with validation"""
        # Input validation
        if not username or not email or not password:
            return False, "All fields are required"

        if len(username) < 3:
            return False, "Username must be at least 3 characters"

        if not validators.email(email):
            return False, "Invalid email format"

        if len(password) < 8:
            return False, "Password must be at least 8 characters"

        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"

        if not re.search(r"\d", password):
            return False, "Password must contain at least one number"

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE username=%s OR email=%s", (username, email))
            if cursor.fetchone():
                cursor.close()
                conn.close()
                return False, "Username or email already exists"

            # Hash password and create user
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING id",
                (username, email, hashed_pw)
            )

            user_id = cursor.fetchone()[0]

            # Create default preferences
            cursor.execute(
                "INSERT INTO user_preferences (user_id) VALUES (%s)",
                (user_id,)
            )

            conn.commit()
            cursor.close()
            conn.close()

            return True, "Registration successful! Please login."

        except Exception as e:
            return False, f"Registration failed: {str(e)}"

    def verify_user(self, username, password):
        """Verify user credentials"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id, password_hash, is_active FROM users WHERE username=%s",
                (username,)
            )
            result = cursor.fetchone()

            if not result:
                cursor.close()
                conn.close()
                return None, "User not found"

            user_id, stored_hash, is_active = result

            if not is_active:
                cursor.close()
                conn.close()
                return None, "Account is deactivated"

            if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE id=%s",
                    (user_id,)
                )
                conn.commit()
                cursor.close()
                conn.close()
                return user_id, "Login successful"
            else:
                cursor.close()
                conn.close()
                return None, "Incorrect password"

        except Exception as e:
            return None, f"Login failed: {str(e)}"

    def get_user_info(self, user_id):
        """Get user information including preferences"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT u.username, u.email, u.created_at, u.last_login,
                       p.theme, p.default_mode, p.email_notifications
                FROM users u
                LEFT JOIN user_preferences p ON u.id = p.user_id
                WHERE u.id = %s
            """, (user_id,))

            result = cursor.fetchone()
            cursor.close()
            conn.close()

            if result:
                return {
                    "id": user_id,
                    "username": result[0],
                    "email": result[1],
                    "created_at": result[2],
                    "last_login": result[3],
                    "theme": result[4] or 'dark',
                    "default_mode": result[5] or 'Beginner',
                    "email_notifications": result[6] if result[6] is not None else True
                }
            return None

        except Exception as e:
            st.error(f"Error getting user info: {str(e)}")
            return None

    def update_user_preferences(self, user_id, theme=None, default_mode=None, email_notifications=None):
        """Update user preferences"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            updates = []
            params = []

            if theme is not None:
                updates.append("theme = %s")
                params.append(theme)
            if default_mode is not None:
                updates.append("default_mode = %s")
                params.append(default_mode)
            if email_notifications is not None:
                updates.append("email_notifications = %s")
                params.append(email_notifications)

            if updates:
                params.append(user_id)
                query = f"UPDATE user_preferences SET {', '.join(updates)} WHERE user_id = %s"
                cursor.execute(query, tuple(params))
                conn.commit()

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            st.error(f"Error updating preferences: {str(e)}")
            return False

    def save_user_portfolio(self, user_id, symbol, quantity, buy_price, buy_date, notes=""):
        """Save portfolio entry for user"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO user_portfolios (user_id, symbol, quantity, buy_price, buy_date, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (user_id, symbol, quantity, buy_price, buy_date, notes))

            conn.commit()
            cursor.close()
            conn.close()
            return True

        except Exception as e:
            st.error(f"Error saving portfolio: {str(e)}")
            return False

    def get_user_portfolio(self, user_id):
        """Get user's portfolio"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, symbol, quantity, buy_price, buy_date, notes
                FROM user_portfolios
                WHERE user_id = %s
                ORDER BY buy_date DESC
            """, (user_id,))

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            portfolio = []
            for row in results:
                portfolio.append({
                    "id": row[0],
                    "symbol": row[1],
                    "quantity": row[2],
                    "buy_price": row[3],
                    "buy_date": row[4],
                    "notes": row[5] or ""
                })

            return portfolio

        except Exception as e:
            st.error(f"Error getting portfolio: {str(e)}")
            return []

    def delete_portfolio_entry(self, user_id, entry_id):
        """Delete a portfolio entry"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM user_portfolios WHERE id = %s AND user_id = %s",
                (entry_id, user_id)
            )

            conn.commit()
            cursor.close()
            conn.close()
            return True

        except Exception as e:
            st.error(f"Error deleting portfolio entry: {str(e)}")
            return False

    def add_to_watchlist(self, user_id, symbol, alert_price=None):
        """Add stock to user's watchlist"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check if already in watchlist
            cursor.execute(
                "SELECT id FROM user_watchlists WHERE user_id = %s AND symbol = %s",
                (user_id, symbol)
            )

            if cursor.fetchone():
                cursor.close()
                conn.close()
                return False, "Stock already in watchlist"

            cursor.execute("""
                INSERT INTO user_watchlists (user_id, symbol, alert_price)
                VALUES (%s, %s, %s)
            """, (user_id, symbol, alert_price))

            conn.commit()
            cursor.close()
            conn.close()
            return True, "Added to watchlist successfully"

        except Exception as e:
            return False, f"Error adding to watchlist: {str(e)}"

    def get_user_watchlist(self, user_id):
        """Get user's watchlist"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, symbol, added_date, alert_price
                FROM user_watchlists
                WHERE user_id = %s
                ORDER BY added_date DESC
            """, (user_id,))

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            watchlist = []
            for row in results:
                watchlist.append({
                    "id": row[0],
                    "symbol": row[1],
                    "added_date": row[2],
                    "alert_price": row[3]
                })

            return watchlist

        except Exception as e:
            st.error(f"Error getting watchlist: {str(e)}")
            return []
