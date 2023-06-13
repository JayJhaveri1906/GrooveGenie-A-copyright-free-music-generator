import pandas as pd

music_genre = pd.read_excel("Data/small_dataset.xlsx", dtype={"track_id":str})
unique_genres = music_genre['genre'].unique()

if __name__ == '__main__':
  for elem in unique_genres:
    embed = get_embedding(elem, tokenizer, bert_model)
    generate_music(elem)

