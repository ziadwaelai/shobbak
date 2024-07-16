import uuid
import psutil
import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
from PIL import Image, UnidentifiedImageError
import re
from transformers import pipeline
from collections import defaultdict
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Function to get MAC address
def get_mac_address():
    mac = uuid.getnode()
    mac_address = ':'.join(('%012X' % mac)[i:i+2] for i in range(0, 12, 2))
    return mac_address

# Database setup
DATABASE_URL = "sqlite:///leaderboard.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class Score(Base):
    __tablename__ = 'scores'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    score = Column(Integer, nullable=False)
    mac_address = Column(String, nullable=False)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Function to get or create user entry
def get_or_create_user():
    mac_address = get_mac_address()
    if mac_address is None:
        st.error("Could not retrieve MAC address. Please check your network settings.")
        return None
    score_entry = session.query(Score).filter_by(mac_address=mac_address).first()
    if score_entry:
        return score_entry
    else:
        name = st.text_input("Enter your name:")
        if st.button("Submit Name"):
            if name:
                new_user = Score(name=name, score=0, mac_address=mac_address)
                session.add(new_user)
                session.commit()
                return new_user
            else:
                st.error("Name cannot be empty")
    return None

# Function to update or create a score entry
def update_score(user, delta):
    user.score += delta
    session.commit()

# Function to get all scores
def get_scores():
    scores = session.query(Score).all()
    return pd.DataFrame([(s.name, s.score) for s in scores], columns=["Name", "Score"])

# Streamlit app layout
st.set_page_config(layout="wide")

st.title("🖼️ PhotoMaster")

# Layout with columns
col1, col2 = st.columns([3, 1])

with col2:
    # Display leaderboard
    st.subheader("Leaderboard")
    df = get_scores()
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # Start index from 1
    df.index.name = 'Rank'
    st.dataframe(df, height=500)
with col1:
    # Get or create user
    user = get_or_create_user()

    if user:
        # Existing functions and logic for processing images
        def convert_drive_link(link):
            match = re.search(r"/d/([^/]+)", link)
            if match:
                file_id = match.group(1)
                return f"https://drive.google.com/uc?export=download&id={file_id}"
            return link

        def download_image(url):
            response = requests.get(url)
            if response.status_code == 200:
                return response.content
            return None

        def resize_image(image_content, size=(1024, 1024)):
            try:
                image = Image.open(BytesIO(image_content))
                image = image.resize(size)
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format="JPEG")
                return img_byte_arr.getvalue()
            except UnidentifiedImageError:
                update_score(user, -1)
                return None

        def remove_background(image_content):
            try:
                image = Image.open(BytesIO(image_content))
                pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
                output_img = pipe(image)
                img_byte_arr = BytesIO()
                output_img.save(img_byte_arr, format="PNG")
                return img_byte_arr.getvalue()
            except UnidentifiedImageError:
                update_score(user, -1)
                return None

        def combine_with_background(foreground_content, background_content, resize_foreground=False):
            try:
                foreground = Image.open(BytesIO(foreground_content)).convert("RGBA")
                background = Image.open(BytesIO(background_content)).convert("RGBA")
                background = background.resize((1024, 1024))

                if resize_foreground:
                    fg_area = foreground.width * foreground.height
                    bg_area = background.width * background.height
                    scale_factor = (0.8 * bg_area / fg_area) ** 0.5

                    new_width = int(foreground.width * scale_factor)
                    new_height = int(foreground.height * scale_factor)

                    foreground = foreground.resize((new_width, new_height))
                    dimensions = (new_width, new_height)
                else:
                    dimensions = (foreground.width, foreground.height)

                fg_width, fg_height = foreground.size
                bg_width, bg_height = background.size
                position = ((bg_width - fg_width) // 2, (bg_height - fg_height) // 2)

                combined = background.copy()
                combined.paste(foreground, position, foreground)
                img_byte_arr = BytesIO()
                combined.save(img_byte_arr, format="PNG")
                return img_byte_arr.getvalue(), dimensions
            except UnidentifiedImageError:
                update_score(user, -1)
                return None, None

        def download_all_images_as_zip(images_info, remove_bg=False, add_bg=False, bg_image=None, resize_foreground=False):
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, "w") as zf:
                for name, url_or_file in images_info:
                    try:
                        if isinstance(url_or_file, str):
                            url = convert_drive_link(url_or_file)
                            image_content = download_image(url)
                        else:
                            image_content = url_or_file.read()

                        if image_content:
                            if remove_bg:
                                processed_image = remove_background(image_content)
                                ext = "png"
                            else:
                                size = (1290, 789) if "banner" in name.lower() else (1024, 1024)
                                processed_image = resize_image(image_content, size=size)
                                ext = "png"

                            if add_bg and bg_image:
                                processed_image, dimensions = combine_with_background(processed_image, bg_image, resize_foreground=resize_foreground)
                                ext = "png"

                            if processed_image:
                                zf.writestr(f"{name.rsplit('.', 1)[0]}.{ext}", processed_image)
                        
                        # Increase score if processed successfully
                        update_score(user, 1)
                    except Exception as e:
                        # Decrease score if there's an error
                        update_score(user, -1)
            zip_buffer.seek(0)
            return zip_buffer

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_files = st.file_uploader("", type=["xlsx", "csv", "jpg", "jpeg", "png", "jfif", "avif", "webp"], accept_multiple_files=True)

        with col2:
            st.markdown("")
            remove_bg = st.checkbox("Remove background")
            add_bg = st.checkbox("Add background")
            resize_fg = st.checkbox("Resize")
            st.checkbox("Compress and Convert Format")
            st.button("Submit")

        images_info = []
        if uploaded_files:
            if len(uploaded_files) == 1 and uploaded_files[0].name.endswith((".xlsx", ".csv")):
                file_type = "excel"
            elif all(file.type.startswith("image/") for file in uploaded_files):
                file_type = "images"
            else:
                st.error("You should work with one type of file: either an Excel file or images.")
                update_score(user, -10)
                file_type = None

            if file_type == "excel":
                uploaded_file = uploaded_files[0]
                if uploaded_file.name.endswith(".xlsx"):
                    xl = pd.ExcelFile(uploaded_file)
                    for sheet_name in xl.sheet_names:
                        df = xl.parse(sheet_name)
                        if "links" in df.columns and "name" in df.columns:
                            df.dropna(subset=["links"], inplace=True)
                            name_count = defaultdict(int)
                            empty_count = 0
                            unique_images_info = []
                            for name, link in zip(df["name"], df["links"]):
                                if pd.isna(name) or name.strip() == "":
                                    empty_name = f"empty_{empty_count}" if empty_count > 0 else "empty"
                                    name = empty_name
                                    empty_count += 1
                                if name_count[name] > 0:
                                    unique_name = f"{name}_{name_count[name]}"
                                else:
                                    unique_name = name
                                unique_images_info.append((unique_name, link))
                                name_count[name] += 1
                            images_info.extend(unique_images_info)
                            if empty_count > 0:
                                st.warning(f"Number of empty cells in 'name' column: {empty_count}")
                                update_score(user, -1)
                        else:
                            st.error(f"The sheet '{sheet_name}' must contain 'links' and 'name' columns.")
                            update_score(user, -10)
                else:
                    df = pd.read_csv(uploaded_file)
                    if "links" in df.columns and "name" in df.columns:
                        df.dropna(subset=["links"], inplace=True)
                        name_count = defaultdict(int)
                        empty_count = 0
                        unique_images_info = []
                        for name, link in zip(df["name"], df["links"]):
                            if pd.isna(name) or name.strip() == "":
                                empty_name = f"empty_{empty_count}" if empty_count > 0 else "empty"
                                name = empty_name
                                empty_count += 1
                            if name_count[name] > 0:
                                unique_name = f"{name}_{name_count[name]}"
                            else:
                                unique_name = name
                            unique_images_info.append((unique_name, link))
                            name_count[name] += 1
                        images_info.extend(unique_images_info)
                        if empty_count > 0:
                            st.warning(f"Number of empty cells in 'name' column: {empty_count}")
                            update_score(user, -1)
                    else:
                        st.error("The uploaded file must contain 'links' and 'name' columns.")
                        update_score(user, -10)
            elif file_type == "images":
                images_info = [(file.name, file) for file in uploaded_files]

        if images_info:
            bg_image = None
            if add_bg:
                bg_file = st.file_uploader("Upload background image", type=["jpg", "jpeg", "png"])
                if bg_file:
                    bg_image = resize_image(bg_file.read())

            st.markdown("## Preview")
            if st.button("Download All Images", key="download_all"):
                try:
                    zip_buffer = download_all_images_as_zip(images_info, remove_bg, add_bg, bg_image, resize_fg)
                    st.download_button(
                        label="Download All Images as ZIP",
                        data=zip_buffer,
                        file_name="all_images.zip",
                        mime="application/zip",
                    )
                    update_score(user, 1)  # Increase score if the whole process is successful
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    update_score(user, -1)  # Decrease score if there's an error

            cols = st.columns(2)
            for i, (name, url_or_file) in enumerate(images_info):
                col = cols[i % 2]
                with col:
                    try:
                        if isinstance(url_or_file, str):
                            url = convert_drive_link(url_or_file)
                            image_content = download_image(url)
                        else:
                            image_content = url_or_file.read()

                        if image_content:
                            if remove_bg:
                                processed_image = remove_background(image_content)
                                ext = "png"
                            else:
                                size = (1290, 789) if "banner" in name.lower() else (1024, 1024)
                                processed_image = resize_image(image_content, size=size)
                                ext = "png"

                            if add_bg and bg_image:
                                processed_image, dimensions = combine_with_background(processed_image, bg_image, resize_foreground=resize_fg)
                                ext = "png"

                            if processed_image:
                                st.image(processed_image, caption=name)
                                st.download_button(
                                    label=f"Download {name.rsplit('.', 1)[0]}",
                                    data=processed_image,
                                    file_name=f"{name.rsplit('.', 1)[0]}.{ext}",
                                    mime=f"image/{ext}",
                                )
                        update_score(user, 1)  # Increase score if image processing is successful
                    except Exception as e:
                        st.error(f"An error occurred while processing {name}: {e}")
                        update_score(user, -1)  # Decrease score if there's an error
