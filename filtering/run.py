from collections import namedtuple, OrderedDict
from timeit import default_timer as timer

import numpy as np
from pandas import DataFrame
from tabulate import tabulate

from thalesians.maths.stats import OnlineMeanAndVarCalculator

class FilterRunData(namedtuple('FilterRunData', (
        'filterrundf',
        'params',
        'stochfilter',
        'duration'))):
    __slots__ = ()
    
    def summary(self):
        s = OrderedDict()
        s['stochastic filter'] = self.stochfilter
        if self.params is not None: 
            for param, value in self.params._asdict().items():
                s['parameter: %s' % param] = value
            s['log-variance theoretical half-life'] = self.params.logvarhalflife()
            s['log-variance theoretical unconditional s.d.'] = np.sqrt(self.params.logvaruncondvar())
        if 'rmse' in self.filterrundf.columns:
            s['rmse'] = self.filterrundf['rmse'].values[-1]
        elif 'truestate' in self.filterrundf.columns and 'posteriorstatemean' in self.filterrundf.columns:
            calculator = OnlineMeanAndVarCalculator()
            calculator.addall(self.filterrundf['truestate'].values - self.filterrundf['posteriorstatemean'].values)
            s['rmse'] = calculator.rms
        if 'loglikelihood' in self.filterrundf.columns:
            s['log-likelihood'] = self.filterrundf['loglikelihood'].values[-1]
        if 'effectivesamplesize' in self.filterrundf.columns:
            s['mean effective sample size'] = np.mean(self.filterrundf['effectivesamplesize'].values)
        s['duration in seconds'] = self.duration
        return s
            
    def __str__(self):
        rows = []
        for name, value in self.summary().items():
            rows.append((name, value))
        return tabulate(rows, headers=('item', 'value'))

def runfilter(df, params, stochfilter, context, observationcolumnname, truestatecolumnname, dropinitialrow=True, observationtransform=None, dtcolumnname=None):
    if dropinitialrow: df.drop(df.index[:1], inplace=True)
    
    columns = ['observation', 'posteriorstatemean', 'posteriorstatevar']
    havetruestate = truestatecolumnname is not None and \
            truestatecolumnname in df.columns and \
            np.any(df[truestatecolumnname].notnull().values)
    if havetruestate:
        columns.append('truestate')
        columns.append('error')
        columns.append('rmse')
        rmsecalculator = OnlineMeanAndVarCalculator()
    if hasattr(stochfilter, 'predictedobservation'): columns.append('predictedobservation')
    if hasattr(stochfilter, 'innov'):
        columns.append('innov')
        if hasattr(stochfilter, 'innovcov'):
            columns.append('innovcov')
            columns.append('standardisedinnov')
    if hasattr(stochfilter, 'gain'):
        columns.append('gain')
        if havetruestate: columns.append('optimalgain')
    if hasattr(stochfilter, 'loglikelihood'):
        columns.append('loglikelihood')
    if hasattr(stochfilter, 'effectivesamplesize'):
        columns.append('effectivesamplesize')
        if hasattr(stochfilter, 'particlecount'):
            columns.append('effectivesamplesizethreshold')
    
    filterrundf = DataFrame(index=df.index, columns=columns)
    filterrundf.fillna(0.0, inplace=True)
    
    start = timer()
    
    for i, row in df.iterrows():
        print(i)
        if dtcolumnname is not None and dtcolumnname in df.columns:
            context['dt'] = row[dtcolumnname]
        else:
            context['dt'] = 1.
        
        stochfilter.predict()
        observation = row[observationcolumnname]
        
        if observationtransform is not None:
            observation = observationtransform(observation, stochfilter)
        
        stochfilter.observe(observation)
        filterrundf['observation'][i] = observation
        m = stochfilter.mean
        if (not np.isscalar(m)) and len(m) > 1: m = m[0,0]
        filterrundf['posteriorstatemean'][i] = m
        v = stochfilter.var
        if (not np.isscalar(v)) and len(v) > 1: v = v[0,0]
        filterrundf['posteriorstatevar'][i] = v
        if havetruestate:
            filterrundf['truestate'][i] = row[truestatecolumnname]
            error = row[truestatecolumnname] - m
            rmsecalculator.add(error)
            filterrundf['error'][i] = error
            filterrundf['rmse'][i] = rmsecalculator.rms
        if hasattr(stochfilter, 'predictedobservation'):
            filterrundf['predictedobservation'][i] = stochfilter.predictedobservation
        if hasattr(stochfilter, 'innov'):
            filterrundf['innov'][i] = stochfilter.innov
            if hasattr(stochfilter, 'innovcov'):
                filterrundf['innovcov'][i] = stochfilter.innovcov
                filterrundf['standardisedinnov'][i] = stochfilter.innov / np.sqrt(stochfilter.innovcov)
        if hasattr(stochfilter, 'gain'):
            g = stochfilter.gain
            if (not np.isscalar(g)) and len(g) > 1: g = g[0,0]
            filterrundf['gain'][i] = g
            if havetruestate:
                filterrundf['optimalgain'][i] = (row[truestatecolumnname] - (m - g * stochfilter.innov)) / stochfilter.innov
        if hasattr(stochfilter, 'loglikelihood'):
            filterrundf['loglikelihood'][i] = stochfilter.loglikelihood
        if hasattr(stochfilter, 'effectivesamplesize'):
            filterrundf['effectivesamplesize'][i] = stochfilter.effectivesamplesize
            if hasattr(stochfilter, 'particlecount'):
                filterrundf['effectivesamplesizethreshold'][i] = 0.5 * float(stochfilter.particlecount)
                
    end = timer()
    
    return FilterRunData(
            filterrundf=filterrundf,
            params=params,
            stochfilter=stochfilter,
            duration = end - start)
