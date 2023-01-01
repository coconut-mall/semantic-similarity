# Imports
import spacy

# Load language model
nlp = spacy.load("en_core_web_md")


# Create a class for Movie objects
class Movie:

    # Def class constructor
    def __init__(self, title, description) -> None:

        # Init variables
        self.title = title
        self.description = description

        # Set default similarity score to 0
        self.similarity_score = 0

    # Def string representation
    def __str__(self) -> str:
        return f"{self.title}"


# Empty list to hold movie objects
movie_list = []


# Open movies file
with open("movies.txt", "r") as movies_file:

    # Loop through each line
    for entry in movies_file:

        # Split the line into a list of title and description
        movie_data = entry.strip().split(" :")

        # Feed that info into the class to create an object
        movie_list.append(Movie(movie_data[0], movie_data[1]))


# Def function which takes in the description of a movie
def movie_recommendation(description_to_compare):

    # Feed the given description into the language model to create a token
    description_to_compare = nlp(description_to_compare)

    # Loop through movie objects
    for movie in movie_list:

        # Feed the movies description into the language model to create token
        movie_description = nlp(movie.description)

        # Compare the given description with the current movies description to get a similarity score
        # Save that score to the object
        movie.similarity_score = movie_description.similarity(
            description_to_compare
        )

    # Get the movie with the highest similarity score
    best_match = max(movie_list, key=lambda movie: movie.similarity_score)

    # Print the best match
    print(
        f"You should watch {best_match} because it had a similarity score of {best_match.similarity_score}"
    )


# Example movie description
plant_hulk = (
    "Will he save the world or destroy it? "
    "When the Hulk becomes too dangerous for the Earth, "
    "The Illuminati trick Hulk into a shuttlew and launch him "
    "into space to a planet where the Hulk can live in peace. "
    "Unfortunately, Hulk lands on the planet Sakaar where he is "
    "sold into slavery and trained as a gladiator."
)


# Run function using the example description
movie_recommendation(plant_hulk)
