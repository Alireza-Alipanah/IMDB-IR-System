from nltk.tokenize import word_tokenize
from preprocess import get_stopwords
import heapq


class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side
        self.stopwords = set(get_stopwords())

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """
        word_tokens = word_tokenize(query)
        filtered_text = [word for word in word_tokens if not word in self.stopwords]
        return ' '.join(filtered_text)

    def find_best_intervals(self, token_occurences):
        intervals = []
        for token, lst in token_occurences.items():
            for i in lst:
                intervals.append((i - self.number_of_words_on_each_side, i + self.number_of_words_on_each_side + 1, token))
        intervals.sort(key=lambda x: (-x[1], -x[0]))
        end = float('inf')
        ans = {}
        for interval in intervals:
            if interval[2] in ans:
                continue
            if interval[1] > end:
                continue
            ans[interval[2]] = interval
            end = interval[0]
        return ans

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []
        doc_tokens = word_tokenize(doc)
        query_tokens = word_tokenize(self.remove_stop_words_from_query(query))
    
        token_occurence = {}
        for token in query_tokens:
            occurrences = [i for i, t in enumerate(doc_tokens) \
                                  if t.lower() == token.lower() and \
                                      i - self.number_of_words_on_each_side >= 0 and i + self.number_of_words_on_each_side + 1 <= len(doc_tokens)]
            occurrences.sort()
            token_occurence[token] = occurrences
        intervals = self.find_best_intervals(token_occurence)
        for token in query_tokens:
            if token not in intervals:
                not_exist_words.append(token)
                continue
            interval = intervals[token]
            window = doc_tokens[interval[0]: interval[1]]
            snippet = ' '.join(window)
            snippet = snippet.replace(token, f"***{token}***")
            final_snippet += snippet + ' ... '

        if final_snippet.endswith(' ... '):
            final_snippet = final_snippet[:-4]
        return final_snippet, not_exist_words
    

if __name__ == '__main__':
    number_of_words_on_each_side = 2
    query = 'test query what'
    doc = 'this query test this is test query for that and just test no meaning'
    snippet = Snippet(number_of_words_on_each_side=number_of_words_on_each_side)
    print(f'query: {query}')
    print(f'doc: {doc}')
    print(f'number_of_words_on_each_side: {number_of_words_on_each_side}')
    print(snippet.find_snippet(doc, query))

