import torch
from collections import defaultdict

printable_method={'transgnn','gat'}

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 11
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, mode='max_acc'):
        result = torch.tensor(self.results[run]).cpu().numpy()
        argmax = result[:, 1].argmax().item()
        argmin = result[:, 3].argmin().item()
        if mode == 'max_acc':
            ind = argmax
        else:
            ind = argmin
        print(f'Run {run + 1:02d}:')
        print(f'Highest Train: {100 * result[:, 0].max():.2f}')
        print(f'Highest Valid: {100 * result[:, 1].max():.2f}')
        print(f'Highest Test: {100 * result[:, 2].max():.2f}')
        print(f'Optimal validation epoch: {ind+1}')
        print(f'Final Train: {100 * result[ind, 0]:.2f}')
        print(f'Final Test: {100 * result[ind, 2]:.2f}')

        print(f"Final acc: {result[ind, 4]:.4f}")
        print(f"Final kappa: {result[ind, 5]:.4f}")
        print(f"Final macro F1: {result[ind, 6]:.4f}")
        print(f"Final micro F1: {result[ind, 7]:.4f}")
        print(f"Final median F1: {result[ind, 8]:.4f}")
        print(f"Final average F1: {result[ind, 9]:.4f}")
        print(f"Final mF1: {result[ind, 10]:.4f}")

        return result[ind]
        

class SimpleLogger(object):
    """ Adapted from https://github.com/CUAI/CorrectAndSmooth """
    def __init__(self, desc, param_names, num_values=2):
        self.results = defaultdict(dict)
        self.param_names = tuple(param_names)
        self.used_args = list()
        self.desc = desc
        self.num_values = num_values
    
    def add_result(self, run, args, values): 
        """Takes run=int, args=tuple, value=tuple(float)"""
        assert(len(args) == len(self.param_names))
        assert(len(values) == self.num_values)
        self.results[run][args] = values
        if args not in self.used_args:
            self.used_args.append(args)
    
    def get_best(self, top_k=1):
        all_results = []
        for args in self.used_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)[-1]
            results_std = results.std(dim=0)

            all_results.append((args, results_mean))
        results = sorted(all_results, key=lambda x: x[1], reverse=True)[:top_k]
        return [i[0] for i in results]
            
    def prettyprint(self, x):
        if isinstance(x, float):
            return '%.2f' % x
        return str(x)
        
    def display(self, args = None):
        
        disp_args = self.used_args if args is None else args
        if len(disp_args) > 1:
            print(f'{self.desc} {self.param_names}, {len(self.results.keys())} runs')
        for args in disp_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)
            results_std = results.std(dim=0)
            res_str = f'{results_mean[0]:.2f} ± {results_std[0]:.2f}'
            for i in range(1, self.num_values):
                res_str += f' -> {results_mean[i]:.2f} ± {results_std[1]:.2f}'
            print(f'Args {[self.prettyprint(x) for x in args]}: {res_str}')
        if len(disp_args) > 1:
            print()
        return results