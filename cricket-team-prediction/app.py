from flask import Flask, render_template, request, jsonify, session, url_for
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import logging
import random

# -------------------- Initialize Flask App --------------------
app = Flask(__name__, static_url_path='/static')
app.secret_key = '23c6a9eb7d6456ae6bf0472cbed52f2d'

# Configure Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.DEBUG)

# -------------------- Sample Teams Data --------------------
INDIA_TEAM = [
    {
        'PlayerName': 'Virat Kohli',
        'Photo': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ0JQfuTAiW8xAbKwCDfK0Eg1_hPB6YjXbtPg&s'
    },
    {
        'PlayerName': 'Rohit Sharma',
        'Photo': 'https://images.news18.com/webstories/uploads/2024/05/WhatsApp-Image-2024-05-07-at-11.48.09_7163f548-2024-05-98e4de156768415486961231099d9f00.jpg'
    },
    {
        'PlayerName': 'Jasprit Bumrah',
        'Photo': 'https://admin.thecricketer.com/weblab/Sites/96c8b790-b593-bfda-0ba4-ecd3a9fdefc2/resources/images/site/bumrahheadshot-min.jpg'
    },
    {
        'PlayerName': 'Md Siraj',
        'Photo': 'https://images.news18.com/ibnlive/uploads/2023/09/mohammed-siraj-wickets-vs-sri-lanka-169495262916x9.jpg?impolicy=website&width=640&height=360'
    },
     {
        'PlayerName': 'Y B Jaiswal',
        'Photo': 'https://media.gettyimages.com/id/2155703459/photo/new-york-new-york-yashasvi-jaiswal-of-india-poses-for-a-portrait-prior-to-the-icc-mens-t20.jpg?s=612x612&w=gi&k=20&c=CflUckAYUHfiGxBoc0y0WLwU4r0q9Og6G4YpjUgWTUc='
    },
     {
        'PlayerName': 'Shreyas Iyer',
        'Photo': 'https://i.pinimg.com/474x/98/2f/a1/982fa1eca0f1f3552daf5983b6fbd1e6.jpg'
    },
     {
        'PlayerName': 'SuryaKumar Yadav',
        'Photo': 'https://w0.peakpx.com/wallpaper/845/347/HD-wallpaper-suryakumar-yadav-suryakumar-sky-surya-batsman-player-india.jpg'
    },
     {
        'PlayerName': 'Axar Patel',
        'Photo': 'https://upload.wikimedia.org/wikipedia/commons/a/ad/Axar_Patel_in_PMO_New_Delhi.jpg'
    },
     {
        'PlayerName': 'Md Shami',
        'Photo': 'https://opt.toiimg.com/recuperator/img/toi/m-69257184/69257184.jpg'
    },
     {
        'PlayerName': 'Rishabh Pant',
        'Photo': 'https://images.cnbctv18.com/wp-content/uploads/2022/09/Rishabh-Pant-in-team-Indias-latest-jersey.jpg?impolicy=website&width=1200&height=900'
    },
    {
        'PlayerName': 'Ravindra Jadeja',
        'Photo': 'https://i.pinimg.com/736x/0e/72/f4/0e72f43c772b278d0ee3ad051239168a.jpg'
    }
]

AUSTRALIA_TEAM = [
    {
        'PlayerName': 'Alex Carey',
        'Photo': 'https://admin.thecricketer.com/weblab/Sites/96c8b790-b593-bfda-0ba4-ecd3a9fdefc2/resources/images/site/careyheadshot-min.jpg'
    },
    {
        'PlayerName': 'Pat Cummins',
        'Photo': 'https://sportsmatik.com/uploads/world-events/players/pat-cummins_1580467882.jpg'
    },
     {
        'PlayerName': 'Josh Hazlewood',
        'Photo': 'https://upload.wikimedia.org/wikipedia/commons/7/77/2018.01.21.17.06.41-Hazelwood_%2839139885264%29.jpg'
    },
     {
        'PlayerName': 'Mitchell Starc',
        'Photo': 'https://cdn.dnaindia.com/sites/default/files/2025/02/27/2679543-mitchell-starc.jpg?im=FitAndFill=(1200,900)'
    },
     {
        'PlayerName': 'Marcus Stoinis',
        'Photo': 'https://upload.wikimedia.org/wikipedia/commons/2/2b/2018.01.21.15.22.25-Stoinis_%2839081521620%29.jpg'
    },
     {
        'PlayerName': 'Sean Abbot',
        'Photo': 'https://static-files.cricket-australia.pulselive.com/headshots/288/1003-camedia.png'
    },
     {
        'PlayerName': 'Mitch Marsh',
        'Photo': 'https://akm-img-a-in.tosshub.com/indiatoday/images/story/202401/mitchell-marsh-rediscovers-lost-intent-to-hit-177-against-bangladesh-courtesy-ap-111837705-3x4.jpg?VersionId=bqbvXxRFWYjVicNzNJJt9YboFiN0Mv2C'
    },
     {
        'PlayerName': 'Steve Smith',
        'Photo': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcReKSwhzE49aiH5gEoScoqvM4YIo4_Z9fM0Zw&s'
    },
    {
        'PlayerName': 'Travis Head',
        'Photo': 'https://d3lzcn6mbbadaf.cloudfront.net/media/details/ANI-20250109085610.jpg'
    },
    {
        'PlayerName': 'Marnus Labuschagne',
        'Photo': 'https://resources.cricket-australia.pulselive.com/photo-resources/2024/11/26/1935628e-00c1-4d06-8448-eeb935f2aabd/marnus-labuschagne-perth-test.jpg?width=1600'
    },
    {
        'PlayerName': 'Josh Inglis',
        'Photo': 'https://images.indianexpress.com/2022/10/Untitled-design-15-4.jpg'
    }
]

# -------------------- Load Data Function --------------------
def load_data(format_type):
    base_path = os.path.join(os.path.dirname(__file__), "datasets")
    file_paths = {
        "Batsmen": os.path.join(base_path, f"Batsman_data_{format_type}.xlsx"),
        "Bowlers": os.path.join(base_path, f"Bowlers_daata_{format_type}.xlsx"),
        "wicketkeeper": os.path.join(base_path, f"WicketKeepers_data_{format_type}.xlsx"),
        "All-Rounders": os.path.join(base_path, f"AllRounder_data_{format_type}.xlsx"),
        "match_details": os.path.join(base_path, f"MatchDetails_{format_type}.xlsx"),
        "playervsplayer": os.path.join(base_path, f"{format_type.lower()}.xlsx")
    }
    data, match_df, pvp_df = {}, None, None
    for role, path in file_paths.items():
        try:
            if os.path.exists(path):
                df = pd.read_excel(path)
                app.logger.debug(f"Loaded {role} data with {len(df)} rows: {df.columns.tolist()}")
                if not df.empty:
                    if role == "match_details":
                        match_df = df
                    elif role == "playervsplayer":
                        pvp_df = df
                    else:
                        data[role] = df
                else:
                    app.logger.warning(f"{role} data is empty: {path}")
            else:
                app.logger.warning(f"File not found: {path}")
        except Exception as e:
            app.logger.error(f"Error loading {role} data: {e}")

    if not data:
        app.logger.error(f"No player data loaded for format {format_type}")
    return data, match_df, pvp_df

# -------------------- Preprocess Data for ML --------------------
def preprocess_data(players_df):
    try:
        if players_df.empty:
            app.logger.warning("players_df is empty")
            return None, None, None
        # Selecting numeric columns except 'points'
        feature_cols = players_df.select_dtypes(include=[np.number]).columns.drop('points', errors='ignore')
        if feature_cols.empty:
            app.logger.warning("No numeric feature columns found, using default points")
            return None, players_df.index, None
        # Extracting feature matrix and target variable
        X = players_df[feature_cols].fillna(0)
        y = players_df['points'].fillna(0) if 'points' in players_df.columns else None
        if y is None or y.isna().all():
            app.logger.warning("No valid points column, using default points")
            return None, players_df.index, None
        return X, y, None  # No scaling applied
    except Exception as e:
        app.logger.exception(f"Error in preprocess_data: {e}")
        return None, None, None

# -------------------- Train ML Models --------------------
def train_decision_tree(X, y):
    try:
        if X is None or y is None:
            app.logger.warning("Invalid data for DecisionTree")
            return None
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y > y.mean())
        return model
    except Exception as e:
        app.logger.exception(f"Error in train_decision_tree: {e}")
        return None

def train_xgboost(X, y):
    try:
        if X is None or y is None:
            app.logger.warning("Invalid data for XGBoost")
            return None
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        model.fit(X, (y > y.mean()).astype(int))
        return model
    except Exception as e:
        app.logger.exception(f"Error in train_xgboost: {e}")
        return None

# -------------------- Predict Teams --------------------
def predict_teams(selected_players, original_players_df):
    try:
        app.logger.debug(f"Predicting with {len(selected_players)} players: {selected_players['PlayerName'].tolist()}")
        if selected_players.empty:
            app.logger.warning("No selected players provided")
            return pd.DataFrame()
        
        selected_players = selected_players.copy()
        # Standardize column names and player names
        selected_players['PlayerName'] = selected_players['PlayerName'].str.strip().str.lower()
        original_players_df.columns = original_players_df.columns.str.strip().str.lower()
        original_players_df['playername'] = original_players_df['playername'].str.strip().str.lower()

        # ------------------- Assigning Points and Roles -------------------
        if 'points' in original_players_df.columns and not original_players_df['points'].isna().all():
            app.logger.debug("Using Points from dataset")
            selected_players['Points'] = selected_players['PlayerName'].map(
                original_players_df.set_index('playername')['points']
            ).fillna(0)
            selected_players['Role'] = selected_players['PlayerName'].map(
                original_players_df.set_index('playername')['role']
            ).fillna('Unknown')
        else:
            app.logger.warning("Points missing, using default points")
            selected_players['Points'] = np.linspace(7.0, 9.0, len(selected_players))
            selected_players['Role'] = 'Unknown'

        # ------------------- Assigning Player Photos -------------------
        # First try to get photos from the original dataset
        if 'photo' in original_players_df.columns:
            original_players_df['photo'] = original_players_df['photo'].fillna('default.jpg')
            photo_mapping = original_players_df.set_index('playername')['photo'].to_dict()
            selected_players['Photo'] = selected_players['PlayerName'].map(photo_mapping)
        else:
            selected_players['Photo'] = None

        # For players without photos, try to match with sample teams
        for idx, row in selected_players.iterrows():
            if pd.isna(row['Photo']) or row['Photo'] == 'default.jpg':
                # Check sample teams for this player
                sample_player = next((p for p in INDIA_TEAM + AUSTRALIA_TEAM 
                                   if p['PlayerName'].lower() == row['PlayerName'].lower()), None)
                if sample_player:
                    selected_players.at[idx, 'Photo'] = sample_player['Photo']
                else:
                    selected_players.at[idx, 'Photo'] = 'images/default.jpg'

        # Final cleanup of photo paths
        selected_players['Photo'] = selected_players['Photo'].str.lower().str.replace(' ', '_')
        selected_players['Photo'] = selected_players['Photo'].apply(
            lambda x: x if x.startswith(('http://', 'https://')) else f'images/{x}'
        )

        app.logger.debug(f"Final player data with photos: {selected_players[['PlayerName', 'Photo']].to_dict(orient='records')}")
        return selected_players

    except Exception as e:
        app.logger.exception(f"Error in predict_teams: {e}")
        return pd.DataFrame()

# -------------------- Generate Balanced Teams --------------------
def generate_balanced_teams(sorted_players, num_teams=4, team_size=11):
    teams = []
    available_players = sorted_players.copy().sort_values(by='Points', ascending=False)
    
    if available_players.empty:
        app.logger.warning("No players provided to generate teams")
        return [pd.DataFrame() for _ in range(num_teams)]

    total_players = len(available_players)
    required_players = num_teams * team_size
    app.logger.debug(f"Total players: {total_players}, required: {required_players}")

    if total_players < required_players:
        app.logger.warning(f"Insufficient players: {total_players} provided, need {required_players}")
        duplicates_needed = (required_players - total_players) // total_players + 1
        available_players = pd.concat([available_players] * duplicates_needed, ignore_index=True)
        available_players = available_players.head(required_players)
        app.logger.debug(f"After duplication, total players: {len(available_players)}")

    full_pool = available_players.copy()

    # Team 1 (India): Prioritize best players
    team1 = create_balanced_team(full_pool.sort_values(by='Points', ascending=False), team_size)
    if not team1.empty:
        teams.append(team1)
        app.logger.debug(f"Team 1 (India) created with {len(team1)} players")
    else:
        teams.append(pd.DataFrame())

    remaining_players = full_pool[~full_pool.index.isin(teams[0].index)].copy()
    for i in range(1, num_teams):
        try:
            if len(remaining_players) >= team_size:
                shuffled_remaining = remaining_players.sample(frac=1, random_state=random.randint(1, 100)).reset_index(drop=True)
                team = create_balanced_team(shuffled_remaining, team_size)
                if not team.empty:
                    total_points = team['Points'].sum()
                    if total_points > 100:
                        team['Points'] = team['Points'] * (100 / total_points)
                    teams.append(team)
                else:
                    teams.append(pd.DataFrame())
            else:
                app.logger.warning(f"Not enough players for Team {i+1}")
                teams.append(pd.DataFrame())
        except Exception as e:
            app.logger.exception(f"Error generating team {i+1}: {e}")
            teams.append(pd.DataFrame())

    while len(teams) < num_teams:
        teams.append(pd.DataFrame())

    return teams

def create_balanced_team(players_df, team_size=11):
    try:
        if len(players_df) < team_size:
            app.logger.warning(f"Not enough players: {len(players_df)} available, need {team_size}")
            return pd.DataFrame()

        selected_players = []
        selected_names = set()
        role_counts = {'Batsmen': 0, 'Bowler': 0, 'wicketkeeper': 0, 'All-Rounder': 0}
        min_requirements = {'Batsmen': 4, 'Bowler': 3, 'wicketkeeper': 1, 'All-Rounder': 3}

        remaining_players = players_df.copy()

        for role, min_num in min_requirements.items():
            while role_counts[role] < min_num and len(selected_players) < team_size:
                candidates = remaining_players[
                    (remaining_players['Role'] == role) & 
                    (~remaining_players['PlayerName'].isin(selected_names))
                ]
                if candidates.empty:
                    app.logger.warning(f"No more {role} players available")
                    break

                top_player = candidates.sample(n=1, random_state=random.randint(1, 100)).iloc[0]
                selected_players.append(top_player.to_dict())
                selected_names.add(top_player['PlayerName'])
                role_counts[role] += 1
                remaining_players = remaining_players[remaining_players.index != top_player.name].copy()

        remaining_slots = team_size - len(selected_players)
        if remaining_slots > 0:
            remaining_players = remaining_players[~remaining_players['PlayerName'].isin(selected_names)]
            if not remaining_players.empty:
                extra_players = remaining_players.sample(n=remaining_slots, random_state=random.randint(1, 100))
                selected_players.extend(extra_players.to_dict('records'))

        team_df = pd.DataFrame(selected_players)[["PlayerName", "Role", "Team", "Points", "Photo"]]
        return team_df
    except Exception as e:
        app.logger.exception(f"Error in create_balanced_team: {e}")
        return pd.DataFrame()

# -------------------- Flask Routes --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/logout")
def logout():
    return render_template("login.html")

@app.route("/profile")
def profile():
    return render_template("profile.html")

@app.route("/team")
def team():
    teams_data = session.get('formatted_teams', [])
    
    # If no teams in session, use sample data
    if not teams_data:
        teams_data = [INDIA_TEAM, AUSTRALIA_TEAM]
        app.logger.debug("Using sample teams data as fallback")
    
    team1_name = session.get('team1_name', "India")
    team2_name = session.get('team2_name', "Australia")
    team3_name = "Team 3"
    team4_name = "Team 4"

    teams = [team_data for team_data in teams_data if team_data]
    return render_template("teamspredicted.html",
                         teams=teams,
                         team1_name=team1_name,
                         team2_name=team2_name,
                         team3_name=team3_name,
                         team4_name=team4_name)

@app.route('/demo-teams')
def demo_teams():
    """Route to demonstrate teams with sample data"""
    return render_template("teamspredicted.html",
                         teams=[INDIA_TEAM, AUSTRALIA_TEAM],
                         team1_name="India",
                         team2_name="Australia",
                         team3_name="Team 3",
                         team4_name="Team 4")

@app.route('/predixi')
def predixi():
    return render_template("predixi.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    app.logger.debug("Entering /predict route")
    try:
        if request.method == "GET":
            return render_template("predixi.html")

        data = request.get_json()
        app.logger.debug(f"Received data: {data}")
        if not data or not data.get("players"):
            app.logger.error("No players provided")
            return jsonify({"error": "No players selected"}), 400

        format_type = data.get("format_type", "ODI")
        venue = data.get("venue")
        players = data.get("players")

        if len(players) != 22:
            app.logger.error(f"Expected 22 players, got {len(players)}")
            return jsonify({"error": "Please provide exactly 22 players (11 per team)"}), 400

        players_data, match_df, pvp_df = load_data(format_type)
        if not players_data:
            app.logger.error("No player data loaded")
            return jsonify({"error": "No player data available"}), 500

        players_df = pd.concat(players_data.values(), ignore_index=True)
        selected_df = players_df[players_df["PlayerName"].isin(players)]
        if len(selected_df) < 22:
            app.logger.warning(f"Only {len(selected_df)} of 22 players found in data")
            return jsonify({"error": "Some players not found in dataset"}), 400

        sorted_players = predict_teams(selected_df, players_df)
        if sorted_players.empty:
            sorted_players = selected_df.copy()
            sorted_players['Points'] = np.linspace(7.0, 9.0, len(selected_df))
            sorted_players['Photo'] = 'images/default.jpg'

        teams = generate_balanced_teams(sorted_players, num_teams=4, team_size=11)
        formatted_teams = [team.to_dict(orient='records') if not team.empty else [] for team in teams]
        session['formatted_teams'] = formatted_teams
        session['team1_name'] = data.get("team1_name", "Team 1")
        session['team2_name'] = data.get("team2_name", "Team 2")

        app.logger.debug(f"Teams generated: {len(formatted_teams)} teams")
        return jsonify({"status": "success"})

    except Exception as e:
        app.logger.exception(f"Error in /predict: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# -------------------- Run Flask App --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
