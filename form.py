import re
import streamlit as st
from datetime import datetime, timedelta
from dateutil import parser

def extract_date_from_query(query):
    # Define a regex pattern to capture various date-related phrases
    date_pattern = r'\b(next\s(?:\w+day|month)|tomorrow|today|\d{1,2}\s(?:\w+)|(?:\w+\s\d{1,2}))\b'
    date_match = re.search(date_pattern, query, re.IGNORECASE)
    
    if date_match:
        date_str = date_match.group(0).strip()

        today = datetime.now()
        if date_str.lower() == 'tomorrow':
            date = today + timedelta(days=1)
        elif date_str.lower() == 'today':
            date = today
        elif 'next month' in date_str.lower():
            # Calculate the first day of the next month
            first_day_next_month = (today.replace(day=1) + timedelta(days=31)).replace(day=1)
            date = first_day_next_month
        else:
            try:
                # Remove 'next ' part from date_str before parsing
                if 'next ' in date_str.lower():
                    date_str = date_str.lower().replace('next ', '')
                
                # Handle month-day and day-month formats
                date = parser.parse(date_str, fuzzy=True)
            except ValueError:
                date = None

        if date:
            return date.strftime('%Y-%m-%d')
        else:
            return None
    else:
        st.write(f"No date found in query: '{query}'")
        return None
    
# Function to handle user appointment booking
def book_appointment(user_query):
    # Extract the date from the user query
    extracted_date = extract_date_from_query(user_query)
    
    if extracted_date:
        st.write(f"It seems like you'd like to book an appointment on **{extracted_date}**.")
        
        name, phone, email = user_detail_form()

        # If valid data is returned, process the appointment
        if name and phone and email:
            st.success(f"Thank you, {name}! Your appointment is scheduled for {extracted_date}.")
            st.write(f"We will contact you shortly at {phone} or via email at {email}.")
    else:
        st.error("Please try again with a valid date format.")


# Function to handle user detail form
def user_detail_form():
    with st.form(key="user_detail_form",clear_on_submit=True):
        name = st.text_input("Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email")

        # Submit button inside form
        submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            # Validate the input details
            if not name or not phone or not email:
                st.error("All fields are required!")
                return None, None, None
            elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("Please enter a valid email address.")
                return None, None, None
            elif not re.match(r"^\+?\d{10}$", phone):
                st.error("Please enter a valid phone number.")
                return None, None, None
            else:
                return name, phone, email

    return None, None, None


if __name__ == "__main__":
    st.write("This is the form.py file.")