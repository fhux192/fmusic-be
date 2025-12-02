import json
import os
import pandas as pd
import plotly.express as px

data_file_path = os.path.join("data", "song_dataset2.jsonl")

if not os.path.exists(data_file_path):
    print(f"Error: File not found at {data_file_path}")
    exit()

print("Reading data...")
songs = []
with open(data_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            songs.append(json.loads(line))

print("Processing groups...")

def get_smart_group(tag_list):
    if not tag_list: return 'Unknown'

    original_tag = tag_list[0]
    tag = original_tag.lower().strip()

    if any(x in tag for x in
           ['sad', 'cry', 'lonely', 'breakup', 'pain', 'regret', 'heartbreak', 'melancholic', 'grief']):
        return 'Sad & Moody'

    if any(x in tag for x in ['happy', 'fun', 'party', 'dance', 'upbeat', 'excited', 'energetic', 'joy']):
        return 'Happy & Energy'

    if any(x in tag for x in ['love', 'romantic', 'sweet', 'crush', 'lovestruck', 'date']):
        return 'Romantic'

    if any(x in tag for x in ['chill', 'peace', 'dreamy', 'relax', 'sleep', 'calm']):
        return 'Chill & Relax'

    if any(x in tag for x in ['intense', 'dramatic', 'angry', 'dark', 'tension']):
        return 'Intense & Drama'

    return original_tag.title()

data = []
for s in songs:
    raw_moods = s.get('mood_tags', [])
    group_name = get_smart_group(raw_moods)

    data.append({
        'Title': s.get('title', 'No Title'),
        'Artist': s.get('artist', 'Unknown'),
        'Group': group_name,
        'Raw_Mood': raw_moods[0] if raw_moods else 'Unknown',
        'Semantic_Snippet': s.get('semantic_text', '')[:100] + '...',
        'Count': 1
    })

df = pd.DataFrame(data)

print("Drawing Treemap...")

fig = px.treemap(
    df,
    path=['Group', 'Artist', 'Title'],
    values='Count',
    color='Group',

    color_discrete_sequence=px.colors.qualitative.Prism,

    hover_data=['Raw_Mood', 'Semantic_Snippet'],
    title='<b>Song Distribution Map (Detailed by Mood)</b>'
)

fig.update_traces(
    root_color="lightgrey",
    marker=dict(cornerradius=3),
    textinfo="label+value"
)

fig.update_layout(
    margin=dict(t=50, l=10, r=10, b=10),
    font=dict(family="Arial", size=14)
)

output_file = "song_treemap_detail.html"
fig.write_html(output_file)
print(f"Done! Output file: {os.path.abspath(output_file)}")