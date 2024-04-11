from typing import List
import math
import wandb


class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        correct_predictions = 0
        total_predictions = 0

        for act, pred in zip(actual, predicted):
            correct_predictions += len(set(act).intersection(set(pred)))
            total_predictions += len(pred)

        precision = correct_predictions / total_predictions if total_predictions > 0 else 0

        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        true_positives = 0
        total_relevant = 0

        for act, pred in zip(actual, predicted):
            true_positives += len(set(act).intersection(set(pred)))
            total_relevant += len(act)

        recall = true_positives / total_relevant if total_relevant > 0 else 0

        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        return f1
    
    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        APs = []
        
        for i in range(len(actual)):
            aps = []
            for j in range(len(predicted[i])):
                if predicted[i][j] in actual[i]:
                    aps.append(self.calculate_precision([actual[i][:j + 1]], [predicted[i][:j + 1]]))

        return aps
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """

        aps = self.calculate_AP(actual, predicted)

        return aps / len(aps)
    
    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0

        for act, pred in zip(actual, predicted):
            relevant = set(act)
            dcg_query = 0.0
            for i, p in enumerate(pred):
                if p in relevant:
                    dcg_query += (2 ** relevant.index(p) - 1) / math.log2(i + 2)
            DCG += dcg_query / len(relevant) if relevant else 0

        return DCG / len(actual)
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        DCG = self.cacluate_DCG(actual, predicted)
        IDCG = 0.0

        for act in actual:
            ideal_prediction = sorted(act, key=lambda x: x in act, reverse=True)
            ideal_DCG = self.cacluate_DCG([act], [ideal_prediction])
            IDCG += ideal_DCG

        IDCG /= len(actual)

        NDCG = DCG / IDCG if IDCG > 0 else 0

        return NDCG
    
    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RRs = []
        for i in range(len(actual)):
            if predicted[i] in actual:
                RRs.append(1 / i)
                continue
        return sum(RRs)
        
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        RRs = []
        for i in range(len(actual)):
            if predicted[i] in actual:
                RRs.append(1 / i)
                continue
        return sum(RRs) / len(RRs)
    

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Average Precision (AP): {ap}")
        print(f"Mean Average Precision (MAP): {map}")
        print(f"Discounted Cumulative Gain (DCG): {dcg}")
        print(f"Normalized Discounted Cumulative Gain (NDCG): {ndcg}")
        print(f"Reciprocal Rank (RR): {rr}")
        print(f"Mean Reciprocal Rank (MRR): {mrr}")
      

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        wandb.log({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'map': map,
            'dcg': dcg,
            'ndcg': ndcg,
            'rr': rr,
            'mrr': mrr,
        })


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)



