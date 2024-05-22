class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()

        for i in range(len(word) - k + 1):
            shingles.add(word[i:i+k])

        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        intersection_size = len(first_set.intersection(second_set))
        union_size = len(first_set.union(second_set))
        if union_size > 0:
            return intersection_size / union_size
        return 0

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        for document in all_documents:
            words = document.split()
            for word in words:
                shingled_word = self.shingle_word(word)
                if word in all_shingled_words:
                    all_shingled_words[word].update(shingled_word)
                else:
                    all_shingled_words[word] = shingled_word
                if word in word_counter:
                    word_counter[word] += 1
                else:
                    word_counter[word] = 1

        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        misspelled_shingles = self.shingle_word(word)
    
        scores_and_words = []
        for candidate_word, candidate_shingles in self.all_shingled_words.items():
            jaccard_score = self.jaccard_score(misspelled_shingles, candidate_shingles)
            scores_and_words.append([jaccard_score, candidate_word])
        scores_and_words.sort(key=lambda x: x[0], reverse=True)
        return scores_and_words[:5]
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        nearest_words = self.find_nearest_words(query)
        tfs = [self.word_counter[word] for score, word in nearest_words]
        max_tf = max(tfs)
        for i in range(len(nearest_words)):
            nearest_words[i][0] *= tfs[i] / max_tf
        return max(nearest_words, key=lambda x: x[0])[1]