# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 21:37:46 2020

@author: Robert
"""
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns

class FactorBacktest(object):
    '''
    因子回测，主要是通过多头，空头和LS组合模拟因子收益率
    '''
    def __init__(self,
                 dataset: dict,
                 parameters: dict):
        '''
        dataset应该包含tradeDates, weight, closeAdj, stockFilter, expectedReturn(值越大资产表现越优秀),benchmark
        以及calcRebalanceDates（可选，如果没有这个key则weight还有expectedReturn必须和tradeDates是对应的）
        所有涉及date的部分都必须是iterable object of yyyymmdd(string date)
        '''
        self.dataset = dataset
        self.parameters = parameters
        
    
    def __mimic_long_return(self,
                            expectedRank: np.array,
                            closeAdj: np.array,
                            weight: np.array,
                            topNumToLong: int,
                            calcRebalanceDates: np.array,
                            tradeableMask: np.array,
                            keepDim = False):
        '''
        给定一个有序指标矩阵(数值越大标的越优异)，循环rebalanceDates中的位置（如果有超出expectedRank中T的值，截断），
        根据tradeableMask给出可交易标的和expectedRank中可交易标的排名最靠前（rank最小）决定
        下一期要long的标的，按照weight归一化之后的权重在第二天买入，直到下一次换仓
        
        Parameters
        -------------------------------------------------
        expectedRank: T/T_1 x N, np.array，典型的例子是预期收益率，或者收益率方向为正的预期因子暴露
        closeAdj: T x N, np.array
        weight: T/T_1 x N, np.array，不要求行和=1，注意，weight矩阵计算中会被calcRebalanceDates mask,expectedRank必须和weight同大小
        topNumToLong: int
        calcRebalanceDates: T_1 x 1, np.array, 是计算换仓的index，实际换仓发生在index+1的位置
        tradeableMask: T x N, boolean np.array
        keepDim: bool
        
        Returns
        -------------------------------------------------
        strategyPnl, strategyTurnover
        
        
        Usage Notes
        -------------------------------------------------
        0. 整个回测基于EOD交易假设，双边费率=0
        1. 当权重数据被load进来之后，第一个处理会是根据当天可交易的股票把当天不可交易的股票的
        权重归为0，然后对权重重新进行归一化
        2. 输入的expectedRank必须不包含当天的值产生的推断，防止未来数据
        3. 当选取被long的股票时，首先会根据当天可交易的股票重新进行排序，根据expectedRank降序排列（越大越优秀）
        选取排名前topNumToLong的股票，这一点意味着如果expectedRank存在大量重复，那么在arrary中位置偏后的股票
        可能会因为topNumToLong的限制而无法被选入
        4. 基于以上假设，一个股票如果在t被买入，t+1需要被调整仓位但不可交易，则股票t+1的weight一定是0
        5. 计算目标权重是基于calcRebalanceDate的权重
        '''
        # 截断rebalanceDates
        calcRebalanceDates = calcRebalanceDates[(calcRebalanceDates<expectedRank.shape[0]) & (calcRebalanceDates>=0)]
        
        # 在weight中仅保留expectedRank靠前的可交易的x个标的
        expectedRank[tradeableMask==0] = -np.inf # 把不能交易的标的的rank标记为inf
        expectedRank = (-1*expectedRank).argsort(axis=1).argsort(axis=1)
        
        # 清洗weight矩阵
        weight[tradeableMask==0] = 0 #去掉不可交易的股票
        weight[expectedRank >= topNumToLong] = 0 # 排名靠后的股票权重为0    
        weight = weight / np.transpose(np.tile(np.sum(weight, axis = 1),(weight.shape[1],1))) # 按照行和为1进行reweight
        if weight.shape[0] == closeAdj.shape[0]:
            weight = weight[calcRebalanceDates,:]
        elif weight.shape[0] == calcRebalanceDates.shape[0]:
            pass
        else:
            raise ValueError('weight必须和closeAdj同大小或者和calcRebalanceDates同大小')
        
        # 初始化pnl
        strategyPnl = np.nan * np.zeros((closeAdj.shape[0]-calcRebalanceDates[0],1))
        strategyTurnover = np.nan * np.zeros((weight.shape[0],1))
        strategyTurnover[0] = 1
        strategyPnl[0] = 1
        k = 1
        lastPanelPnl = 1
        rd_weight = weight[1,:] # target weight on rebalance date
        
        # 循环每个截面
        for panelIdx in range(len(calcRebalanceDates)):
            if panelIdx < calcRebalanceDates.shape[0] - 1:
                rebalanceDate = calcRebalanceDates[panelIdx] + 1
                nextRebalanceDate = calcRebalanceDates[panelIdx+1] + 1
            else:
                rebalanceDate = min(calcRebalanceDates[panelIdx] + 1, calcRebalanceDates[-1])
                nextRebalanceDate = closeAdj.shape[0]-1
                
            rd_closeAdj = closeAdj[rebalanceDate,:] # rebalance date close adj
            rd_toLongAssets = weight[panelIdx,:] > 0 # get boolean asset to long
            
            for holdDates in range(rebalanceDate + 1, nextRebalanceDate + 1):
                strategyPnl[k] = lastPanelPnl * np.sum(closeAdj[holdDates, rd_toLongAssets]/rd_closeAdj[rd_toLongAssets]*rd_weight[rd_toLongAssets])
                k = k + 1
            
            lastPanelPnl = strategyPnl[k-1] # 指向下一个调仓日
            
            if panelIdx < calcRebalanceDates.shape[0] - 1:
                nrd_weight = weight[panelIdx+1,:] # 下一个调仓日的期望权重
                nrd_cantClearAssets = (nrd_weight==0) & (rd_weight>0) & (tradeableMask[nextRebalanceDate,:] == 0) # 在下一个调仓日不能被清空的资产（因为当天不能交易的股票的权重都是0）
                nrd_rd_weight = closeAdj[nextRebalanceDate,:]/closeAdj[rebalanceDate,:]*rd_weight # 从这个调仓日到下一个调仓日时标的的实际权重
                nrd_rd_weight[rd_toLongAssets] = nrd_rd_weight[rd_toLongAssets] / np.sum(nrd_rd_weight[rd_toLongAssets]) # 权重归一化
                nrd_rd_weight[rd_toLongAssets==0] = 0
                
                if np.sum(nrd_cantClearAssets) != 0: # 如果有不能被清仓的股票
                    nrd_rd_weight = closeAdj[nextRebalanceDate,:]/closeAdj[rebalanceDate,:]*rd_weight
                    nrd_weight = np.zeros(1, expectedRank.shape[1])
                    nrd_weight[nrd_cantClearAssets] = nrd_rd_weight[nrd_cantClearAssets] / np.sum(nrd_rd_weight[rd_toLongAssets]) #计算不能被清仓的股票占比
                    nrd_weight[nrd_weight>0] = (1-np.sum(nrd_weight[nrd_cantClearAssets]))*nrd_weight[nrd_weight>0]
                else:
                    nrd_weight = weight[panelIdx+1,:]
                    
                rd_weight = nrd_weight
                strategyTurnover[panelIdx+1] = np.sum(np.abs(nrd_weight-nrd_rd_weight))
        
        if keepDim:
            return np.concatenate((np.nan*np.ones((calcRebalanceDates[0],1)),strategyPnl),axis=0), strategyTurnover
        else:
            return strategyPnl, strategyTurnover


    def track_long_portfolio(self, numAssetsToLong: int, rebalanceFreq: str, numYearDays = 252):
        '''
        追踪策略多头收益

        Parameters
        ---------------------------------------
        numAssetsToLong: int, 每个截面要long的资产个数
        rebalanceFreq: str, 有'week','month'

        Returns
        --------------------------------------
        strategyPnl, strategyPerf
        '''
        weight = self.dataset['weight']
        tradeDates = self.dataset['tradeDates']
        closeAdj = self.dataset['closeAdj']
        stockFilter = self.dataset['stockFilter']
        expectedReturn = self.dataset['expectedReturn']
        benchmark = self.dataset['benchmark']
        startDate = self.parameters['startDate']
        endDate = self.parameters['endDate']

        try:
            calcRebalanceDates = self.dataset['calcRebalanceDates']
        except KeyError:
            calcRebalanceDates = tradeDates

        # 根据startDate和endDate对tradeDates, closeAdj, benchmark,stockFilter, expectedReturn,weight
        # 进行裁切
        startLoc = list(np.isin(tradeDates, np.array([startDate])).index(True))
        endLoc = list(np.isin(tradeDates, np.array([endDate])).index(True))
        tradeDates = tradeDates[startLoc:endLoc+1]
        closeAdj = closeAdj[startLoc:endLoc+1,:]
        benchmark = benchmark[startLoc:endLoc+1,:]
        benchmark = benchmark / benchmark[0]
        stockFilter = stockFilter[startLoc:endLoc+1,:]

        startLocRd = list(np.isin(calcRebalanceDates, np.array([startDate])).index(True))
        endLocRd = list(np.isin(calcRebalanceDates, np.array([endDate])).index(True))
        calcRebalanceDates = calcRebalanceDates[startLocRd:endLocRd+1]
        weight = weight[startLocRd:endLocRd+1,:]
        expectedReturn = weight[startLocRd:endLocRd+1,:]

        _, dateLoc = self.extract_date_from_trade_dates(calcRebalanceDates, freq = rebalanceFreq)
        weight = weight[dateLoc,:]
        expectedReturn = expectedReturn[dateLoc,:]

        strategyPnl, strategyTurnover = self.__mimic_long_return(expectedReturn,
                                                                 closeAdj,
                                                                 weight,
                                                                 numAssetsToLong,
                                                                 dateLoc,
                                                                 stockFilter,
                                                                 keepDim = True)
        performance = self.calc_performance(strategyPnl,
                                            benchmark,
                                            numYearDays,
                                            strategyTurnover)
        return {'tradeDates':tradeDates,
                'strategyPnl':strategyPnl,
                'benchmarkPnl':benchmark,
                'performance':performance}


    def extract_date_from_trade_dates(self, tradeDates, freq = 'week'):
        '''
        Parameters
        ---------------------------------------
        tradeDates: iterable object of string datetime(YYYYmmdd)
        freq: str, 'week' or 'month' or 'year'

        Returns
        ---------------------------------------
        extractDates: np.array of string datetime(YYYYmmdd)
        extractLoc: np.array， 返回extractDates在tradeDates中的location
        '''
        extractDates = []
        extractLoc = []
        for dateIdx in range(len(tradeDates)-1):
            aDate = tradeDates[dateIdx]
            nextDate = tradeDates[dateIdx+1]
            dtObj = datetime.strptime(aDate, '%Y%m%d')
            nextDtObj = datetime.strptime(nextDate, '%Y%m%d')

            if freq == 'week':
                dtWeek = dtObj.isocalendar()[1]
                nextDtWeek = nextDtObj.isoformat()[1]
                if nextDtWeek > dtWeek:
                    extractDates.append(aDate)
                    extractLoc.append(dateIdx)
            elif freq == 'month':
                dtMonth = dtObj.month
                nextDtMonth = nextDtObj.month
                if nextDtMonth > dtMonth:
                    extractDates.append(aDate)
                    extractLoc.append(dateIdx)
            elif freq == 'year':
                dtYear = dtObj.year
                nextDtYear = nextDtObj.year
                if nextDtYear > dtYear:
                    extractDates.append(aDate)
                    extractLoc.append(dateIdx)
            else:
                raise ValueError("freq参数值可以是'week', 'month'或者'year'")

        return extractDates, extractLoc


    def calc_performance(self, strategyPnl, benchmarkPnl, numYearDays, strategyTurnover = None):
        '''
        分析策略的表现，包含，年化收益率，年化波动率，SR，最大回撤，年化超额收益，跟踪误差，信息比率，超额最大回撤，
        平均换手

        Returns
        ----------------------------------------
        pd.DataFrame, columns = ['年化收益率',
                                '年化波动率',
                                '夏普比率',
                                '最大回撤',
                                '年化累计超额收益',
                                '跟踪误差',
                                '信息比率',
                                '超额收益最大回撤',
                                '平均双边换手率']
        '''
        annualReturn = self.calc_annualized_return(strategyPnl, numYearDays)
        annualVol = self.calc_annualize_volatility(strategyPnl, numYearDays)
        sharpeRatio = self.calc_sharpe_ratio(strategyPnl)
        maxDrawdown = self.calc_max_drawdown(strategyPnl)
        annualExcessiveReturn = self.calc_annualized_excessive_return(strategyPnl,benchmarkPnl,numYearDays)
        annualTrackError = self.calc_annualized_track_error(strategyPnl,benchmarkPnl,numYearDays)
        icir = annualExcessiveReturn / annualTrackError
        excessiveReturnMaxDrawdown = self.calc_max_drawdown(self.calc_excessive_return(strategyPnl, benchmarkPnl))

        if strategyTurnover is not None:
            avgTurnover = np.mean(strategyTurnover[1:]) # 第一次换不算
        else:
            avgTurnover = np.nan

        return pd.DataFrame(data = [annualReturn, annualVol, sharpeRatio, maxDrawdown,
                                    annualExcessiveReturn, annualTrackError,icir, excessiveReturnMaxDrawdown,
                                    avgTurnover],
                            columns = ['年化收益率','年化波动率','夏普比率','最大回撤',
                                       '年化累计超额收益','跟踪误差','信息比率','超额收益最大回撤',
                                       '平均双边换手率'])

        
    @staticmethod
    def calc_drawdown(strategyPnl):
        '''
        计算策略回撤
        '''
        ts = np.nan * np.zeros(strategyPnl.shape)
        ts[0] = 0
        for timeId in range(1, strategyPnl.shape[0]):
            ts[timeId] = np.abs(1 - strategyPnl[timeId]/np.max(strategyPnl[:timeId]))
        
        return ts


    @staticmethod
    def calc_annualized_return(strategyPnl, numYearDays = 250):
        '''
        计算年化收益, 注意，由于基于EOD交易假设， t0到t0+1实际只过了1天
        '''
        return np.power(strategyPnl[-1]/strategyPnl[0],numYearDays/(len(strategyPnl)-1))-1
        
    
    @staticmethod
    def calc_annualize_volatility(strategyPnl, numYearDays = 250):
        '''
        计算年化波动率
        '''
        return np.std(strategyPnl[1:]/strategyPnl[:-1], ddof=1)*np.sqrt(numYearDays)


    @staticmethod
    def calc_excessive_return(strategyPnl, benchmarkPnl):
        '''
        计算累计超额收益,base = 1
        '''
        ts = np.nan * np.zeros(strategyPnl.shape)
        ts[0] = 1
        for timeId in range(1, strategyPnl.shape[0]):
            ts[timeId] = ts[timeId - 1] * (
                        strategyPnl[timeId] / strategyPnl[timeId - 1] - benchmarkPnl[timeId] / benchmarkPnl[timeId - 1])

        return ts


    def calc_sharpe_ratio(self, strategyPnl):
        '''
        计算sharpe比率
        '''
        return self.calc_annualized_return(strategyPnl)/self.calc_annualize_volatility(strategyPnl)
    
    
    def calc_max_drawdown(self, strategyPnl):
        '''
        计算最大回撤
        '''
        return np.max(self.calc_drawdown(strategyPnl))
    
    
    def calc_annualized_excessive_return(self, strategyPnl, benchmarkPnl, numYearDays = 250):
        '''
        计算年化超额收益
        '''
        return self.calc_annualized_return(self.calc_excessive_return(strategyPnl, benchmarkPnl), numYearDays)


    def calc_annualized_track_error(self, strategyPnl, benchmarkPnl, numYearDays = 250):
        '''
        计算跟踪误差
        '''
        return np.std(strategyPnl[1:]/strategyPnl[:-1]-benchmarkPnl[1:]/benchmarkPnl[:-1],ddof=1)*np.sqrt(numYearDays)
    
    
    def plot_report(self, longReturn, shortReturn, benchmarkReturn, tradeDates):
        """
        生成追踪报告
        """
        data = pd.DataFrame(columns = ['tradeDate','多头收益','空头收益','benchmark收益']，
                            data = [tradeDates, longReturn, shortReturn, benchmarkReturn])
        