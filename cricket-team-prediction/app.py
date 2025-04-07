from flask import Flask, render_template, request, jsonify, session, url_for
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import logging
import random
# -------------------- Initialize Flask App --------------------
app = Flask(_name_,static_url_path='/static', template_folder='templates')
app.secret_key = '23c6a9eb7d6456ae6bf0472cbed52f2d'
# Configure Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.DEBUG)

# -------------------- Load Data Function --------------------
def load_data(format_type):
    base_path = os.path.join(os.path.dirname(_file_), "datasets")
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
        # Standardize column names and player names (to avoid case-sensitive mismatches)
        selected_players['PlayerName'] = selected_players['PlayerName'].str.strip().str.lower()
        original_players_df.columns = original_players_df.columns.str.strip().str.lower()
        original_players_df['playername'] = original_players_df['playername'].str.strip().str.lower()

        # ------------------- Assigning Points and Roles -------------------
        if 'points' in original_players_df.columns and not original_players_df['points'].isna().all():
            app.logger.debug("Using Points from dataset")

            # Map existing points from original dataset
            selected_players['Points'] = selected_players['PlayerName'].map(
                original_players_df.set_index('playername')['points']
            ).fillna(0)

            # Map roles from original dataset
            selected_players['Role'] = selected_players['PlayerName'].map(
                original_players_df.set_index('playername')['role']
            ).fillna('Unknown')  # Default to 'Unknown' if role is not found
        else:
            app.logger.warning("Points missing, using default points")
            selected_players['Points'] = np.linspace(7.0, 9.0, len(selected_players))  # Default points if empty
            selected_players['Role'] = 'Unknown'

        # ------------------- Assigning Player Photos -------------------
        if 'photo' in original_players_df.columns:
            original_players_df['photo'] = original_players_df['photo'].fillna('default.jpg')

            # Create a mapping of lowercase player names to their photo filenames
            photo_mapping = original_players_df.set_index('playername')['photo'].to_dict()
            app.logger.debug(f"Photo Mapping: {photo_mapping}")  # Debugging

            # Apply the mapping to selected players
            selected_players['Photo'] = selected_players['PlayerName'].map(photo_mapping).fillna('images/default.jpg')

            # Convert filenames to lowercase and replace spaces with underscores (for consistency)
            selected_players['Photo'] = selected_players['Photo'].str.lower().str.replace(' ', '_')

            # Prepend 'images/' to match the path expected by teamspredicted.html
            selected_players['Photo'] = 'images/' + selected_players['Photo']
        else:
            selected_players['Photo'] = 'images/default.jpg'
            app.logger.warning("No 'Photo' column in dataset, using default.jpg")

        # Debugging Output
        app.logger.debug(f"Final Assigned Data: {selected_players[['PlayerName', 'Role', 'Points', 'Photo']].to_dict(orient='records')}")
        print("Predicted Players and Photos:")
        print(selected_players[['PlayerName', 'Role', 'Points', 'Photo']].to_string(index=False))

        return selected_players

    except Exception as e:
        app.logger.exception(f"Error in predict_teams: {e}")
        return pd.DataFrame()

# -------------------- Generate Balanced Teams --------------------
def generate_balanced_teams(sorted_players, num_teams=4, team_size=11):
    teams = []
    available_players = sorted_players.copy().sort_values(by='Points', ascending=False)  # Sort by points for priority
    
    if available_players.empty:
        app.logger.warning("No players provided to generate teams")
        return [pd.DataFrame() for _ in range(num_teams)]

    total_players = len(available_players)
    required_players = num_teams * team_size
    app.logger.debug(f"Total players: {total_players}, required: {required_players}")

    # Duplicate players to meet the required number (44 for 4 teams of 11)
    if total_players < required_players:
        app.logger.warning(f"Insufficient players: {total_players} provided, need {required_players}. Duplicating players.")
        duplicates_needed = (required_players - total_players) // total_players + 1
        available_players = pd.concat([available_players] * duplicates_needed, ignore_index=True)
        available_players = available_players.head(required_players)  # Trim to exactly 44 players
        app.logger.debug(f"After duplication, total players: {len(available_players)}")

    # Keep the full pool for each team, but prioritize best players for the first team
    full_pool = available_players.copy()

    # Team 1 (India): Prioritize best players
    team1 = create_balanced_team(full_pool.sort_values(by='Points', ascending=False), team_size)
    if not team1.empty:
        teams.append(team1)
        app.logger.debug(f"Team 1 (India) created with {len(team1)} players: {team1['PlayerName'].tolist()}")
    else:
        teams.append(pd.DataFrame())
        app.logger.debug(f"Team 1 (India) is empty")

    # Subsequent teams: Use remaining players with reduced priority
    remaining_players = full_pool[~full_pool.index.isin(teams[0].index)].copy()
    for i in range(1, num_teams):
        try:
            if len(remaining_players) >= team_size:
                # Shuffle remaining players and select based on reduced priority
                shuffled_remaining = remaining_players.sample(frac=1, random_state=random.randint(1, 100)).reset_index(drop=True)
                team = create_balanced_team(shuffled_remaining, team_size)
                if not team.empty:
                    total_points = team['Points'].sum()
                    if total_points > 100:
                        team['Points'] = team['Points'] * (100 / total_points)
                    teams.append(team)
                    app.logger.debug(f"Team {i+1} created with {len(team)} players: {team['PlayerName'].tolist()}")
                else:
                    teams.append(pd.DataFrame())
                    app.logger.debug(f"Team {i+1} is empty")
            else:
                app.logger.warning(f"Not enough players for Team {i+1}: {len(remaining_players)} remaining")
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
        selected_names = set()  # Track player names to avoid duplicates within the team
        role_counts = {'Batsmen': 0, 'Bowler': 0, 'wicketkeeper': 0, 'All-Rounder': 0}
        min_requirements = {'Batsmen': 4, 'Bowler': 3, 'wicketkeeper': 1, 'All-Rounder': 3}

        # Create a copy of the DataFrame to modify
        remaining_players = players_df.copy()

        # Fill roles according to minimum requirements
        for role, min_num in min_requirements.items():
            while role_counts[role] < min_num and len(selected_players) < team_size:
                # Filter players for the current role, excluding already selected players
                candidates = remaining_players[
                    (remaining_players['Role'] == role) & 
                    (~remaining_players['PlayerName'].isin(selected_names))
                ]
                if candidates.empty:
                    app.logger.warning(f"No more {role} players available (need {min_num}, have {role_counts[role]})")
                    break

                # Select a random player from the candidates to ensure diversity
                top_player = candidates.sample(n=1, random_state=random.randint(1, 100)).iloc[0]
                selected_players.append(top_player.to_dict())
                selected_names.add(top_player['PlayerName'])
                role_counts[role] += 1
                # Remove the selected player from remaining_players
                remaining_players = remaining_players[remaining_players.index != top_player.name].copy()

        # Fill remaining slots with any available players
        remaining_slots = team_size - len(selected_players)
        if remaining_slots > 0:
            remaining_players = remaining_players[~remaining_players['PlayerName'].isin(selected_names)]
            if not remaining_players.empty:
                extra_players = remaining_players.sample(n=remaining_slots, random_state=random.randint(1, 100))
                selected_players.extend(extra_players.to_dict('records'))
                selected_names.update(extra_players['PlayerName'].tolist())

        team_df = pd.DataFrame(selected_players)[["PlayerName", "Role", "Team", "Points", "Photo"]]
        app.logger.debug(f"Team created: {team_df.to_dict(orient='records')}")
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
    team1_name = session.get('team1_name', "Team 1")
    team2_name = session.get('team2_name', "Team 2")
    team3_name = "Team 3"
    team4_name = "Team 4"

    teams = [team_data for team_data in teams_data if team_data]
    app.logger.debug(f"Rendering {len(teams)} teams: {[len(t) for t in teams]}")
    return render_template("teamspredicted.html",
                          teams=teams,
                          team1_name=team1_name,
                          team2_name=team2_name,
                          team3_name=team3_name,
                          team4_name=team4_name)

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
            sorted_players['Points'] = np.linspace(7.0, 9.0, len(selected_df))  # Default points if empty
            sorted_players['Photo'] = 'images/default.jpg'

        teams = generate_balanced_teams(sorted_players, num_teams=4, team_size=11)
        formatted_teams = [team.to_dict(orient='records') if not team.empty else [] for team in teams]
        session['formatted_teams'] = formatted_teams
        session['team1_name'] = data.get("team1_name", "Team 1")
        session['team2_name'] = data.get("team2_name", "Team 2")

        app.logger.debug(f"Teams generated: {len(formatted_teams)} teams")
        for i, team in enumerate(formatted_teams):
            app.logger.debug(f"Team {i+1}: {team}")

        return jsonify({"status": "success"})

    except Exception as e:
        app.logger.exception(f"Error in /predict: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# -------------------- Run Flask App --------------------
if _name_ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=false)
