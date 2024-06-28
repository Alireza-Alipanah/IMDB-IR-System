########## needed for relative import ##########
import inspect
import sys
import os

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
################################################

try:
    from graph import LinkGraph
    from indexer.indexes_enum import Indexes
    from indexer.index_reader import Index_reader
except Exception:
    pass

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = {}
        self.authorities = {}
        self.move_titles = {}
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            if movie['stars'] is None:
                continue
            self.hubs[movie['id']] = 1
            self.graph.add_node(movie['id'])
            for star in movie['stars']:
                self.graph.add_node(star)
                self.graph.add_edge(movie['id'], star)
                self.graph.add_edge(star, movie['id'])
                self.authorities[star] = 1
            self.move_titles[movie['id']] = movie['title']
            

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            if movie['stars'] is None:
                continue
            self.graph.add_node(movie['id'])
            for star in movie['stars']:
                self.graph.add_node(star)
                self.graph.add_edge(movie['id'], star)
                self.graph.add_edge(star, movie['id'])
                
    def normalize(self, d):
        s = sum(d.values())
        for i in d.keys():
            d[i] = d[i] / s

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        for i in range(num_iteration):
            old_hub_scores = self.hubs.copy()
            old_authorities_scores = self.authorities.copy()
            for hub in self.hubs.keys():
                score = sum([old_authorities_scores[i] for i in self.graph.get_successors(hub, self.authorities.keys())])
                self.hubs[hub] = score
            for authority in self.authorities.keys():
                score = sum([old_hub_scores[i] for i in self.graph.get_successors(authority, self.hubs.keys())])
                self.authorities[authority] = score
            self.normalize(self.authorities)
            self.normalize(self.hubs)
            
        top_actors = sorted(self.authorities, key=self.authorities.get, reverse=True)[:max_result]
        top_movies = sorted(self.hubs, key=self.hubs.get, reverse=True)[:max_result]
        top_movies = [self.move_titles[movie_id] for movie_id in top_movies]
        return top_actors, top_movies
        
            

if __name__ == "__main__":
    def get_root_set_keys():
        from search import SearchEngine
        search_engine = SearchEngine()
        query = "spider man in wonderland"
        method = "Unigram"
        smoothing_method = 'mixture'
        weights = {
            Indexes.STARS: 1,
            Indexes.GENRES: 1,
            Indexes.SUMMARIES: 1
        }
        result = search_engine.search(query, method, weights, smoothing_method=smoothing_method, max_results=50)
        return set([res[0] for res in result])
    
    
    def get_corpus():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'indexer', 'index')
        return Index_reader(path, Indexes.DOCUMENTS).index
    
    # You can use this section to run and test the results of your link analyzer
    root_keys = get_root_set_keys()
    indexes = get_corpus()
    corpus = [{'id': indexes[i]['id'], 'stars': indexes[i]['stars']} for i in indexes.keys() if i not in root_keys]
    root_set = [{'id': indexes[i]['id'], 'stars': indexes[i]['stars'], 'title': indexes[i]['title']} for i in root_keys]

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
