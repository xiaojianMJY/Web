package com.boom.weka.classfier;


import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.RemoveRange;
/**
 * 通过C4.5训练电费数据
 * @author qdl
 * @since 10/12
 */
public class Classfier {

	public Classfier() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) {
		int i;
		try {			
			/*
			ArffLoader loader=new ArffLoader();
			loader.setFile(new File("./data/tab1.arff"));
			Instances data=loader.getDataSet();
			*/
			/**
			 * 读取数据
			 */
			Instances data=DataSource.read("D:\\DataMining\\实验数据_实验1_郭琨\\tab1.csv");
			
			/**
			 * 使用最后一个属性作为类别属性
			 */
			if(data.classIndex()==-1)
				data.setClassIndex(data.numAttributes()-1);
			
			/**
			 * 将数值型属性规范化至[0,1]区间
			 * 默认参数
			 * -S 1.0 -T 0.0
			 */
			Normalize filter=new Normalize();
			//所有设置必须在设置格式之前
			filter.setInputFormat(data);
			Instances newData=Filter.useFilter(data,filter);		
		
			
			/**
			 * 将数值型属性转换为标签型属性
			 */
			NumericToNominal filterNtoN=new NumericToNominal();
			Instances result=null;
			result=new Instances(newData);
			filterNtoN.setInputFormat(result);
			//filterStoN.setAttributeRange("last");
			result=Filter.useFilter(result, filterNtoN);
		
			
			/**
			 * 选择属性
			 * InfoGainAttributeEval
			 * ranker:选取前五个属性
			 */
			
			System.out.println(result.numAttributes());
			AttributeSelection attsel=new AttributeSelection();
			InfoGainAttributeEval infoEval=new InfoGainAttributeEval();
			Ranker searchOfRanker=new Ranker();
			//选取前五个属性
			searchOfRanker.setNumToSelect(5);
			attsel.setEvaluator(infoEval);
			attsel.setSearch(searchOfRanker);
			attsel.SelectAttributes(result);
			System.out.println(attsel.toResultsString());
			//编号减1，选取前五个属性
			int[] indices=attsel.selectedAttributes();
			
			/**
			 * 删除部分属性
			 */
			Remove rmAttr=new Remove();
			String [] rmOption=new String[3];
			rmOption[0]="-R";
			rmOption[1]=new Integer(indices[0]+1).toString();
			for(i=1;i<5;i++)
			{
				rmOption[1]+=","+(indices[i]+1);
			}
			rmOption[1]+=",last";
			System.out.println(rmOption[1]);
			rmOption[2]="-V";
			rmAttr.setOptions(rmOption);
			rmAttr.setInputFormat(result);
			result=Filter.useFilter(result, rmAttr);
			//System.out.println(result.toSummaryString());
			
			/**
			 * 离散化
			 */
			Discretize disAttr=new Discretize();
			disAttr.setInputFormat(result);
			result=Filter.useFilter(result, disAttr);
			
			/**
			 * 按比例获取部分实例
			 */
				RemovePercentage rmPer=new RemovePercentage();
				String []rmPOption=new String[2];
				rmPOption[0]="-P";
				rmPOption[1]="70";
				rmPer.setOptions(rmPOption);
				rmPer.setInputFormat(result);
				result=Filter.useFilter(result, rmPer);
				
				
			/**
			 * 按比例获取训练和测试数据
			 * 训练数据=60%
			 * 测试数据=40%
			 */
			RemoveRange removeR=new RemoveRange();
			String [] options=new String[2];
			int mid=(result.numInstances()-1)*6/10;		
			options[0]="-R";
			options[1]="first-"+(mid+1); 
		//	System.out.println(options[0]);
			removeR.setOptions(options);
			removeR.setInputFormat(result);
			Instances test=Filter.useFilter(result, removeR);
		//	System.out.println(test.numInstances());			
			options[0]="-R";
			options[1]=(mid)+"-last"; 
		//	System.out.println(options[0]);
			removeR.setOptions(options);
			removeR.setInputFormat(result);
			Instances train=Filter.useFilter(result, removeR);
		//	System.out.println(train.numInstances());
			
			
			/*
			for(i=0;i<result.numAttributes()-1;i++)
			{//数据处理
				if(result.attribute(i).isString())
				{
				System.out.println(i);	
					StringToNominal filterStoN=new StringToNominal();
					filterStoN.setInputFormat(result);
					result=Filter.useFilter(result, filterStoN);
				}
			}*/
			/**
			 * 输出属性信息
			 */
			/*for(i=0;i<result.numAttributes()-1;i++)
				System.out.println(result.attribute(i));*/
			

			
			
			/**
			 * J48分类器
			 * 训练数据=60%
			 * 建立分类器
			 */
			
			/**
			 * 评估器evaluate
			 * cross-validation--十字交叉
			 * dedicated test set--指定测试集
			 * 选用指定测试集进行评估
			 */

			
			J48 tree=new J48();
		
			tree.buildClassifier(train);
			
			 Evaluation evalTree=new Evaluation(train);
			 evalTree.evaluateModel(tree, test);
			 System.out.println("\n决策树运行结果\n");
			 System.out.println(evalTree.toClassDetailsString("\n=====Class detail=====\n\n"));
			 System.out.println(evalTree.toSummaryString("\n=====Results=====\n\n",false));
			 System.out.println(evalTree.toMatrixString("\n=====Confusion Matrix=====\n\n"));
			 System.out.println("\n=====Time Cost=====\n");
			 //未知
			 System.out.println(evalTree.totalCost());

			 
			 /**
			  * 随机森林分类器
			  */
			 
			
			 RandomForest randForest=new RandomForest();	
			 randForest.buildClassifier(train);			 
			 Evaluation evalForest=new Evaluation(train);
			 evalForest.evaluateModel(randForest, test);
			 System.out.println("\n随机森林运行结果\n");
			 System.out.println(evalForest.toClassDetailsString("\n=====Class detail=====\n\n"));
			 System.out.println(evalForest.toSummaryString("\n=====Results=====\n\n",false));
			 System.out.println(evalForest.toMatrixString("\n=====Confusion Matrix=====\n\n"));
			 System.out.println("\n=====Time Cost=====\n");
			 //未知
			 System.out.println(evalForest.totalCost());
			 
			 /**
			  * 贝叶斯分类
			  */
			 
			 NaiveBayes navieBayes=new NaiveBayes();			 
			 navieBayes.buildClassifier(train);
			 Evaluation evalBayes=new Evaluation(train);
			 evalBayes.evaluateModel(navieBayes, test);
			 System.out.println("\n贝叶斯运行结果\n");
			 System.out.println(evalBayes.toClassDetailsString("\n=====Class detail=====\n\n"));
			 System.out.println(evalBayes.toSummaryString("\n=====Results=====\n\n",false));
			 System.out.println(evalBayes.toMatrixString("\n=====Confusion Matrix=====\n\n"));
			 System.out.println("\n=====Time Cost=====\n");
			 //未知
			 System.out.println(evalBayes.totalCost());
			 
			 /**
			  * 神经网络分类
			  * 
			  */
			 

			 rmPOption[0]="-P";
			 rmPOption[1]="98";
			 rmPer.setOptions(rmPOption);
			 rmPer.setInputFormat(train);
			 train=Filter.useFilter(train, rmPer);
			 rmPer.setInputFormat(test);
			 test=Filter.useFilter(test, rmPer);
			// System.out.println(test.numInstances()+" "+train.numInstances());	
				
				
			 MultilayerPerceptron bp=new MultilayerPerceptron(); 
			 bp.buildClassifier(train);			 
			 Evaluation evalBp=new Evaluation(train);
			 evalBp.evaluateModel(bp, test);
			 System.out.println("\n神经网络运行结果\n");
			 System.out.println(evalBp.toClassDetailsString("\n=====Class detail=====\n\n"));
			 System.out.println(evalBp.toSummaryString("\n=====Results=====\n\n",false));
			 System.out.println(evalBp.toMatrixString("\n=====Confusion Matrix=====\n\n"));
			 System.out.println("\n=====Time Cost=====\n");
			 //未知
			 System.out.println(evalBp.totalCost());
			 
			 
		
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

}
