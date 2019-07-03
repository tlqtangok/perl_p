//#include "common.hpp"
/*
 * jd create at 2018-02-28
 * to build & run on Linux, just run :
 * make
 */
//#pragma once
#pragma warning(disable:4996)
#pragma warning(disable:4018)

#include <iostream>
#include <fstream>

#include <string>
#include <sstream>
#include <vector>
#include <stdint.h>  // uint64_t
#include <assert.h>
#include <list>

#include <vector>
#include <sstream>
#include <unordered_map>
#include <map>
#include <algorithm>    // std::sort
#include <utility>

#include <stack>
#include <memory.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <thread>
#include <mutex>


// test 
#include <tuple>
#include <type_traits>
#include <utility>
#include <future>
#include <omp.h>
#include <array>
#include <numeric>

#if __linux__
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <htslib/faidx.h>
#include <strings.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include "htslib/tbx.h"
#include "htslib/sam.h"
#include "htslib/vcf.h"
#include "htslib/kseq.h"
#include "htslib/bgzf.h"
#include "htslib/hts.h"
#include "htslib/regidx.h"
#include <unordered_map>
#include <iomanip>
#include <list>
//#include "hfile_internal.h"
#include <unistd.h>


//#include "samtools.h"
//#include "version.h"
//#include <config.h>
#endif


#ifdef __linux__
#define __P__  return 0;   //__
#else
#define __P__  system("pause");return 0;   //__
#define popen(fp, fr) _popen((fp),(fr))
#define pclose(fp) _pclose(fp)
#define sleep(i) Sleep(1000*(i))

#endif
using namespace std;


#define DEL_ARR_MEM(P) if(NULL != (P)){delete [] (P); (P) = NULL;}



#if 0

#define DEBUG 0 //DEBUG=1, use debug version, vice versa.
#define FLAG_USE_MULTI_THREAD 1 //thr_, FLAG_USE_MULTI_THREAD = 0, single thread; FLAG_USE_MULTI_THREAD = 1, MultiThread.
#define E_BATCH_NUM_BR 3500 //e_batch_num_br = 10000
#define TD_NUM_BR 4 //td_num_br = 20
#define E_BATCH_NUM_PR 4000 //e_batch_num_pr = 10000
#define TD_NUM_PR 4 //td_num_pr = 20

#define AUTOTEST 0 
#define br_Ana 1 //br_Ana = 1, pr_Ana = 0, only run br; br_Ana = 1, pr_Ana = 1, both no run.
#define pr_Ana 0 //br_Ana = 0, pr_Ana = 1, only run pr; br_Ana = 0, pr_Ana = 0, both run.
#define dq_adv 1 //dq_adv = 1, advanced computing of deltaQ
#define should_have_argc 13
#define CacheChromosome 1 //CacheChromosome = 1, then load one chromosome in memory and reuse it, else then not.



class sam_record;
class bed_record;
class grp_tables;

using namespace std;
// jd add frm 2018-02-27

/*
   . get sam read

   . get ref
   . get vcf files
   . get bed files
   . gen grp tables
   . gen grp table Arguments
   . gen grp table Quantized
   . gen grp table RecalTable0
   . gen grp table RecalTable1
   . gen grp table RecalTable2
   */

class cml_parser
{
	public:
		map<string, vector<string>> map_;
		cml_parser(int argc_, char** argv_)
		{
			static vector<string> keys_list{ "-knownSites",  "-R", "-I", "-o", "-L", "-grp", "-grpPath" };
			vector<string> vec_arg{};

			auto len = argc_;
			for (int i = 0; i < len; i++)
			{
				vec_arg.push_back(string(argv_[i]));
			}

			for (auto &e_key : keys_list)
			{
				vector<string>::iterator it = vec_arg.begin();
				while ((it = std::find(it, vec_arg.end(), e_key)) != vec_arg.end())
				{
					it++;
					map_[e_key].push_back(*it);
					//cout << *it << endl;
				}
			}
		};

		int has_opt(string e_key)
		{
			return map_.find(e_key) == map_.end() ? 0 : 1;
		}
		vector<string> get_arg_opt(string e_key)
		{
			assert(has_opt(e_key));
			return map_[e_key];
		}
};


// ****** headers, ordered *****
//#include "bgitools.hpp"
namespace bgitools
{
	struct SAMSequenceRecord
	{
		int mSequenceIndex;
		int mSequenceLength;
	};

	struct Block
	{
		long startPosition;
		long size;
		Block()
		{
			startPosition = -1;
		}
		Block(long &startPos, long & sz)
		{
			startPosition = startPos;
			size = sz;
		}
	};

	struct ChrIndex
	{
		string name;
		int binWidth;
		int longestFeature;
		int nFeatures;
		vector<Block> blocks;
		bool OLD_V3_INDEX = false;
		void read(char * &in, char * &is, ifstream & _if, const int & maxSize);
		Block getBlocks(int & start, int & end);
	};

	enum OPT
	{
		N, //Skipped region from the reference
		H, //Hard clip
		P, //Padding
		D, //Deletion vs. the reference
		I, //Insertion vs. the reference
		S, //Soft clip
		M, //Match or mismatch
		EQ, //Matches the reference
		X //Mismatches the reference
	};

	struct CIGAR
	{
		int length;
		OPT opt; //opt: M,I,D,etc...
	};

	struct CigarOperator
	{
		bool consumesReferenceBases;
		bool consumesReadBases;
		char character;
	};

	enum EventType
	{
		BASE_SUBSTITUTION,
		BASE_INSERTION,
		BASE_DELETION
	};

	string fcout(const string & fn, const string & fc, const std::ios::openmode & F_W = (ios::out | ios::trunc)); // fc: file content
	double round_(double val, int precision);
	string join_vec_2_str(vector<string>& vec_string, const char & delimiter = '\t');

	vector<string> split_str_2_vec(const string & str, const char & delimiter = '\t', const int & retainNum = -1, int flag_push_tail_str = 0);
	void trim_str(char * str);
	void trim_str(string & str);
	int run_cmd(const char* cmd, char *res);
	template<typename T> string to_string(const T &t);
	template<typename TK, typename TV>
		std::vector<TK> extract_keys(std::unordered_map<TK, TV> const& input_map);
	bool isRegularBase(const char & base);
	bool isLowQualityBase(sam_record& e_sam_record, const int & offset);
	int getAlignmentEnd(sam_record & e_sam_record, const bool & cacheCigar = true);

	string safeGetRefSeq(sam_record & e_sam_record, faidx_t *fai);
	string safeGetRefSeq_v1(sam_record & e_sam_record, string & chromosome);
	vector<pair<int, int>> getFeaturesByHtslib_v2(string & chr_start_end, string & fn_vcf, int & n_vcf);
	map<int, vector<pair<int, int>>*> getMapFeatures(string & rname, int & start, int & end, vector<string> & vec_fn_vcf);

	int readInt(char * &in);
	string readString(char * &in);
	long readLong(char * &in, char *& is, ifstream & _if, const int & maxSize);
	void readSequenceDictionary(char * &in);
	vector<pair<string, string>> readHeader(char * &in, char *& is, ifstream & _if, const int & maxSize);
	vector<pair<string, bgitools::SAMSequenceRecord>> getSequenceDictionaryFromProperties(vector<pair<string, string>> & properties);
	string getFeatureReader(string & inputFile);
#if 0
	pair<vector<pair<string, string>>, unordered_map<string, ChrIndex>> getFeatureSource(string & inputFile);
#endif 

#if 0
	vector<unordered_map<string, ChrIndex>> loadVcfIdx(vector<string> & vec_fn_vcf);
#endif 

#if 0
	string vcfReadLine(char * &fin, char *& fis, ifstream & _fif, const int & maxSize);
#endif 

#if 0
	string vcfDecode(char * &fin, char *& fis, ifstream & _fif, const int & maxSize);
#endif 
#if 0
	map<int, vector<pair<int, int>>> getFeaturesByVcfIdx(string & rname, int & start, int & end, vector<string> & vec_fn_vcf);
#endif 
#if 0
	vector<pair<int, int>> getVecFeaturesByVcfIdx(string & rname, int & start, int & end, string & fn_vcf, int & n_vcf);
#endif 

	bool basesAreEqual(char & base1, char & bases2);
	unordered_map<int, struct CIGAR> decode_cigar(string & cigar);
	unordered_map<int, struct CIGAR> get_cigar(sam_record & e_sam_record);
	struct CigarOperator cigarOperator(bgitools::OPT & opt);
	bool MalformedReadFilter(sam_record & e_sam_record);

	vector<bed_record> read_bed_file_2_vec_bed_record(const string &fn);
	unordered_map<string, vector<bed_record>> vec_bed_record_2_map(vector<bed_record> & vec_bed_record);
	bool bed_filter_v1(sam_record& e_sam_record, vector<bed_record> & sorted_vec_interval);
	bool sam_filter(sam_record& e_sam_record);
};



//#include "globalb.hpp"
namespace globalB
{
#pragma pack(1)
	struct vcf_range
	{
		uint64_t pos;
		uint16_t len;
	};
	struct chr_start_len
	{
		uint64_t offset;
		uint64_t len;
	};
#pragma pack()

	struct vcfRelatedPointer
	{
		const char * fname;
		htsFile *fp;
		tbx_t *tbx = NULL;
	};

	struct hg19Pointer
	{
		faidx_t *fai;
	};

	mutex g_mutex;
	const uint32_t cb = 7;      // GATK:band width [7]
	const double gapOpenPenalty = 40; // GATK:public static final double DEFAULT_GOP = 40;
	const double ce = 0.1;    // GATK:gap extension probability [0.1]
	const int MAX_PHRED_SCORE = 93;
	const int minBaseQual = 4;
	const double EM = 0.33333333333;
	const double EI = 0.25;
	const int MAX_VALUE = 127;
	const int READ_PAIRED_FLAG = 0x1;
	const int READ_UNMAPPED_FLAG = 0x4;
	const int READ_STRAND_FLAG = 0x10;
	const int SECOND_OF_PAIR_FLAG = 0x80;
	const int mismatchesContextSize = 2;
	const int LENGTH_BITS = 4;

	const int MAXIMUM_CYCLE_VALUE = 500; // GATK:the maximum cycle value permitted for the Cycle covariate
	const char LOW_QUAL_TAIL = 2;
	const int LENGTH_MASK = 15;
	const int QUANTIZING_LEVELS = 16; // GATK:number of distinct quality scores in the quantized output
	const int SMOOTHING_CONSTANT = 1; //GATK:used when calculating empirical qualities to avoid division by zero
	const int CLIPPING_GOAL_NOT_REACHED = -1;
	const int MIN_USABLE_Q_SCORE = 6;

	const string STANDARD_INDEX_EXTENSION = ".idx";
	const double MAX_FEATURES_PER_BIN = 100;
	const int MAX_BIN_WIDTH = 1 * 1000 * 1000 * 1000; //  GATK:widths must be less than 1 billion
	const long MAX_BIN_WIDTH_FOR_OCCUPIED_CHR_INDEX = 1024000;
	const int SEQUENCE_DICTIONARY_FLAG = 0x8000; // GATK:if we have a sequence dictionary in our header
	const string SequenceDictionaryPropertyPredicate = "DICT:";
	const char LINEFEED = { '\n' & 0xff };
	const char CARRIAGE_RETURN = { '\r' & 0xff };
	const int BUFFER_OVERFLOW_INCREASE_FACTOR = 2;
	const char HEADER_INDICATOR = '#';

	vector<string> vec_chr;
	double *qual2prob = NULL;
	double ***EPSILONS = NULL;//EPSILONS = [256][256][MAX_PHRED_SCORE+1]

	double globalDeltaQ;
	unordered_map<int, double> map_deltaQReported;
	double ***deltaQCovariatesArr = NULL;
	unordered_map<int, int> context_v_k_Map;
	unordered_map<int, int> cycle_v_k_Map;

	void initializeCachedData(double *** array3d, double *array1d);
	double convertFromPhredScale(const double & x);
	int createMask(const int & contextSize);
	vector<int> contextWith(string & bases, const int & contextSize, const int & mask);
	vector<double> qualToErrorProb();
	vector<struct vcfRelatedPointer> vec_vcfPointer;

	grp_tables *grp_table_arr = NULL;
	void grp_allocate(int & td_num);
	void grp_release(int & td_num);

	const int GrpRT1LENGTH = 100;
	const int GrpRT2SHAPE_dim_0 = 2;
	const int GrpRT2SHAPE_dim_1 = 256;
	const int GrpRT2SHAPE_dim_2 = GrpRT2SHAPE_dim_1;

	const vector<int> eps_3d_shape = { 256, 256, MAX_PHRED_SCORE + 1 }; //TODO:some day 256 may not applicable to readLength greater than 50.
	void allocate_3d_epsilons_1d_qual2prob();
	void relatedDeltaQInit(grp_tables & gt, int & readLength);
	void release_3d_epsilons_1d_qual2prob();
	void relatedDeltaQRelease();

	const double cd = globalB::convertFromPhredScale(globalB::gapOpenPenalty);  // GATK:gap open probability [1e-3]
	const int mismatchesKeyMask = globalB::createMask(mismatchesContextSize);
	const vector<double> qualToErrorProbCache = globalB::qualToErrorProb();

	string chromosome;
	void load_chromosome(string & rname, faidx_t *fai);


	// load vcf.bin and vcf.bin.idx
	FILE *if_vcf_bin = NULL;
	map<string, chr_start_len> map_chr_range{};
	vcf_range *arr_vcf_range_R0 = NULL;
	int sz_arr_vcf_range_R0 = 0;

	vector<vcf_range*> arr_vcf_range{};
	vector<int> arr_sz_vcf_range{};


	map<string, chr_start_len>& init_vcf_idx_2_map(string& fn_vcf, uint64_t &max_len);
	void alloc_bytes_of_arr_vcf_range(uint64_t &max_len);
	void delete_bytes_of_arr_vcf_range();
	int load_vcf_range_by_chrname(string &search_chrname);
	int bin_search_nearest_smaller_idx(globalB::vcf_range* arr_vcf_range, int& sz, uint64_t& s_start, uint64_t &last_s_start);
	void vcf_record_uniq();
	int getFeaturesByHtslib(vcf_range* arr_vcf_range, int &start, int &end, vector<globalB::vcf_range*>& vec_overlap_p_range);
}



//#include "samtools.hpp"
namespace samtools
{
	enum ClippingRepresentation
	{
		/** GATK:Clipped bases are changed to Ns */
		WRITE_NS,

		/** GATK:Clipped bases are changed to have Q0 quality score */
		WRITE_Q0S,

		/** Clipped bases are change to have both an N base and a Q0 quality score */
		WRITE_NS_Q0S,

		/**
		 * GATK:Change the read's cigar string to soft clip (S, see sam-spec) away the bases.
		 * Note that this can only be applied to cases where the clipped bases occur
		 * at the start or end of a read.
		 */
		SOFTCLIP_BASES,

		/**
		 * GATK:WARNING: THIS OPTION IS STILL UNDER DEVELOPMENT AND IS NOT SUPPORTED.
		 *
		 * Change the read's cigar string to hard clip (H, see sam-spec) away the bases.
		 * Hard clipping, unlike soft clipping, actually removes bases from the read,
		 * reducing the resulting file's size but introducing an irrevesible (i.e.,
		 * lossy) operation.  Note that this can only be applied to cases where the clipped
		 * bases occur at the start or end of a read.
		 */
		HARDCLIP_BASES,

		/**
		 * GATK:Turn all soft-clipped bases into matches
		 */
		REVERT_SOFTCLIPPED_BASES,
	};

	enum ClippingTail
	{
		LEFT_TAIL,
		RIGHT_TAIL
	};

	struct CigarShift
	{
		unordered_map<int, struct bgitools::CIGAR> cigar_map;
		int shiftFromStart;
		int shiftFromEnd;
	};

	pair<int, int> calculateQueryRange(sam_record& e_sam_record); //ensure: two elements
	uint32_t getInsertionOffset(sam_record& e_sam_record, const int & ind);

	map<string, vector<sam_record*>> read_sam_file_2_map_vec_p_sam_record(const string & fn);
	void del_map_vec_p_sam_record(map<string, vector<sam_record *> >& map_chr_vec_p_sam_record);

	bool getReadNegativeStrandFlag(sam_record& e_sam_record);
	string simpleReverseComplement(string & bases);
	bool getReadPairedFlag(const int & flag);
	bool getSecondOfPairFlag(const int & flag);
	int keyFromCycle(int & cycle);
	int keyFromContext(const string & bases, const int & start, const int & end);
	int simpleBaseToBaseIndex(const char & base);
	string baseIndexToSimpleBase(int & baseIndex);
	sam_record clipLowQualEnds(sam_record & e_sam_record, const char & lowQual, const samtools::ClippingRepresentation & algorithm);

	int getAdaptorBoundary(sam_record & e_sam_record);
	bool getReadUnmappedFlag(sam_record & e_sam_record);
	void hardClipByReferenceCoordinates(sam_record & e_sam_record, int & refStart, int & refStop);
	pair<bool, struct bgitools::CIGAR> readStartsWithInsertion(unordered_map<int, struct bgitools::CIGAR> & cigar_map);
	int getReadLengthForCigar(unordered_map<int, struct bgitools::CIGAR> & cigar_map);
	int getSoftStart(unordered_map<int, struct bgitools::CIGAR> & cigar_map, uint32_t & alignmentStart);
	int getSoftEnd(unordered_map<int, struct bgitools::CIGAR> & cigar_map, uint32_t & alignmentEnd);
	int calculateHardClippingAlignmentShift(struct bgitools::CIGAR & cigarElement, const uint32_t & clippedLength);
	int calculateAlignmentStartShift(unordered_map<int, struct bgitools::CIGAR> & old_cigar_map, unordered_map<int, struct bgitools::CIGAR> & new_cigar_map);
	struct CigarShift cleanHardClippedCigar(unordered_map<int, struct bgitools::CIGAR> & cigar_map);
	struct CigarShift hardClipCigar(sam_record & e_sam_record, int & start, int & stop);
	void hardClip(sam_record & e_sam_record, int & start, int & stop);
	void hardClipAdaptorSequence(sam_record & e_sam_record);
	void clipRead(sam_record & e_sam_record, vector<pair<int, int>> & op);
	void hardClipSoftClippedBases(sam_record & e_sam_record);

	string sam_record_to_string(map<string, vector<sam_record*>>& vec_sam_record);
};



//#include "grptools.hpp"
namespace grptools
{
	struct RT2KEY
	{
		int quality;
		int covariateKey;
		int covariateIdx;
	};

	class Arguments
	{
		public:
			Arguments();
			string to_string();
			string binary_tag_name;
			string covariate;
			string default_platform;
			string force_platform;

			string indels_context_size;
			string insertions_default_quality;
			string low_quality_tail;
			string mismatches_context_size;

			string mismatches_default_quality;
			string no_standard_covs;
			string plot_pdf_file;
			string quantizing_levels;

			string recalibration_report;
			string run_without_dbsnp;
			string solid_nocall_strategy;
			string solid_recal_mode;
	};

	struct QualityScore
	{
		int count;
		int quantizedScore;
		string to_string(const long &key, const char* record_format, char* e_line, const int & MAX_CHAR_E_LINE);
	};

	struct RecalTable0
	{
		double empiricalQuality;
		double estimatedQReported;
		double observations;
		double numMismatches; // Errors
		RecalTable0()
		{
			observations = 0;
			empiricalQuality = -1;
			estimatedQReported = -1;
			numMismatches = 0;
		}
		string to_string(string& ReadGroup, string &EventType, const char* record_format, char* e_line, const int & MAX_CHAR_E_LINE);
	};

	struct RecalTable1
	{
		double empiricalQuality;
		double observations;
		double numMismatches; // Errors

		double estimatedQReported; // no need this in output str
		RecalTable1()
		{
			observations = 0;
			empiricalQuality = -1;
			numMismatches = 0;
		}
		string to_string(string& ReadGroup, int &key, string &EventType, const char* record_format, char* e_line, const int & MAX_CHAR_E_LINE);
	};


	struct RecalTable2
	{
		double empiricalQuality;
		double observations;
		double numMismatches;
		RecalTable2()
		{
			observations = 0;
			empiricalQuality = -1;
			numMismatches = 0;
		}
		string to_string(string &ReadGroup, RT2KEY &key, string &EventType, const char* record_format, char* e_line, const int & MAX_CHAR_E_LINE);
	};

	struct QualInterval
	{
		int qStart;
		int qEnd;
		int fixedQual;
		int level;
		long nObservations;
		long nErrors;
		int mergeOrder;
		QualInterval()
		{
			qStart = -1;
		}

		vector<struct grptools::QualInterval> subIntervals;
	};

	struct CVKEY
	{
		int quality = 0;
		int contextKey = 0;
		int cycleKey = 0;
	};

	struct grpTempDat
	{   // skip,snpErrors,readCovariates has same length
		struct grptools::CVKEY *readCovariates;
		double *snpErrors;
		bool *skip;
		int length = 0;
	};

	//TODO:add knownSites read method
	vector<bool> calculateSkipArray(sam_record& e_sam_record, vector<bool> & knownSites);
	void calculateSkipArray_v1(sam_record& e_sam_record, vector<bool> & knownSites, grptools::grpTempDat * grpTempDatArr, int & idx);
	vector<int> calculateIsSNP(sam_record& e_sam_record, string & refSeq);
	void calculateAndStoreErrorsInBlock(int iii, int & blockStartIndex, vector<int> & errorArray, vector<double> & fractionalErrors);
	void calculateAndStoreErrorsInBlock_v1(int iii, int & blockStartIndex, vector<int> & errorArray, double * fractionalErrors);
	vector<double> calculateFractionalErrorArray(vector<int> & isSNP, string & baqArray);
	void calculateFractionalErrorArray_v1(vector<int> & isSNP, string & baqArray, grptools::grpTempDat * grpTempDatArr, int & idx);
	unordered_map<int, struct CVKEY> ComputeCovariates(sam_record & e_sam_record);
	vector<struct CVKEY> ComputeCovariates_v1(sam_record & e_sam_record);
	void ComputeCovariates_v2(sam_record & e_sam_record, grptools::grpTempDat * grpTempDatArr, int & idx);
	double calcExpectedErrors(const double & estimatedQReported, const double & observations);
	string formatKey(int & key);
	string contextFromKey(int & key);
	double getEmpiricalQuality(double & empiricalQuality, double & observations, double & numMismatches);
	int probToQual(long double & prob, long double & eps);
	double getErrorRate(long & nObservations, long & nErrors, int & fixedQual);
	struct grptools::QualInterval merge(struct grptools::QualInterval & fromMerge, struct grptools::QualInterval & toMerge);
	double getPenalty(struct grptools::QualInterval & interval, double & globalErrorRate);
	vector<struct grptools::QualInterval> removeAndAdd(vector<struct grptools::QualInterval> & interval, struct grptools::QualInterval & qualInterval);
	pair<int, bool>  getReadCoordinateForReferenceCoordinate(int & alignmentStart, unordered_map<int, struct bgitools::CIGAR> & cigar_map,
			int & refCoord, const samtools::ClippingTail & tail, bool & allowGoalNotReached);
	int getReadCoordinateForReferenceCoordinate(sam_record & e_sam_record, int & refCoord, const samtools::ClippingTail & tail, bool & allowGoalNotReached);
	vector<pair<int, int>> getBindings(sam_record & e_sam_record, vector<globalB::vcf_range*> & e_vec_vcf_overlap_p_range, vector<pair<int, int>> &vec_ft_it, std::vector<std::list<int>> & currentFeatures);


	vector<bool> calculateKnownSitesByFeatures(sam_record & e_sam_record, vector<pair<int, int>> & bindings);

	vector<sam_record*> filterPreprocess_v1(int & start, int & end, vector<sam_record>& vec_line_sam_record, vector<bed_record> & vec_bed_record);
	vector<sam_record*> aggregateReadData(vector<sam_record*> & vec_p_sam_record, string &rname, vector<globalB::vcf_range*>& e_vec_vcf_overlap_p_range, string &hg19_path, string& fn_vcf);
}



//#include "bqsrtools.hpp"
namespace bqsrtools
{
	void grp_reduce(grp_tables & gt, grp_tables * grp_tables_arr, int & td_num);

	double CalculateDeltaQCovariates(struct grptools::RecalTable2 *** frt2, vector<int> & keyVec, double & globalDeltaQ, double & deltaQReported);
	double CalculateDeltaQReported(struct grptools::RecalTable1 * rt1, double & globalDeltaQ, int & qualFromRead);
	double calculateGlobalDeltaQ(struct grptools::RecalTable0 & rt0);
	int PerformSequentialQualityCalculation(grp_tables & gt, vector<int> & keySet, bgitools::EventType & errorModel);
}



//#include "baqtools.hpp"
namespace baqtools
{
	class BAQRESULT
	{
		public:
			string refBases;
			string readBases;
			uint32_t queryStart;
			uint32_t queryLen;
			string rawQuals;
			char *bq = NULL;
			int *state = NULL;
			BAQRESULT(int sz_bq = 0, int sz_state = 0);
			~BAQRESULT();
	};

	char capBaseByBAQ(char & oq, char & bq, int & state, int & expectedPos);
	int set_u(const int& b, const int& i, const int& k);
	void hmm_glocal(string & ref, string & query, uint32_t & qstart, uint32_t & l_query, string & _iqual, char *bq, int *state, int &bqsize);
	string calcBAQFromHMM(sam_record& e_sam_record, string & refSeq);
	string getBAQTag(sam_record& e_sam_record, string & refSeq);

	const int SZ_BAQ_M = 9;
	double baqtools_m[SZ_BAQ_M] = { 0 };
}



//#include "sam_record.hpp"
class sam_record
{
	public:
		sam_record() { }
		sam_record(const string& e_line);
		~sam_record() {}

		string to_string();

		static string header_str;
		static string group_name;
		static const int VEC_SZ;

		string qname; // 1th
		uint16_t flag;
		string rname;
		uint32_t pos;
		uint32_t alignmentEnd = -1;

		uint8_t mapq;
		string cigar;
		unordered_map<int, struct bgitools::CIGAR> cigar_map;
		vector<pair<int, int>> bindings;
		string refSeq;
		string rnext;
		uint32_t pnext;

		int32_t tlen;
		string seq;
		string qual;
		static string BI;
		static string BD; // 13th, extra field will be: 23 - 13 = 10
		string ex_str;  //
	private:
		string& ex_str_to_string();
};



//#include "bed_record.hpp"
class bed_record
{
	public:
		~bed_record() {};
		bed_record(const string& e_line);

		string chrname;
		uint32_t start;
		uint32_t end;
		//string snp_name;
		//string snp_type;
};



//#include "grp_tables.hpp"
class grp_tables
{
	public:
		grp_tables() { assert(NULL == e_line); e_line = new char[MAX_CHAR_E_LINE]; assert(NULL != e_line); };
		~grp_tables() { DEL_ARR_MEM(e_line); };
		void updateDataForRead(unordered_map<int, struct grptools::CVKEY> & readCovariates, vector<bool> & skip, vector<double> & snpErrors);
		void updateDataForRead_v1(vector<struct grptools::CVKEY> & readCovariates, vector<bool> & skip, vector<double> & snpErrors);
		void updateDataForRead_v2(grptools::grpTempDat * grpTempDatArr, int & size);
		void updateDataForRead_v3(grptools::grpTempDat * grpTempDatArr, int & size);
		void Quantized_option();
		vector<struct grptools::QualInterval> quantize(vector<long> & qualHistogram);
		void recalTable0_option(int & qual, double & isError);
		void RecalTable1_option(int & key, double & isError);
		void RecalTable2_option(grptools::CVKEY & key, double & isError, int & covariateIndex);
		void allocateRT2();
		void releaseRT2();
		string eventType;

		const int  MAX_CHAR_E_LINE = 115 * 2;
		char *e_line = NULL;

		grptools::Arguments arguments;
		string arguments_to_string();
		string arguments_fcout(const string &fn = "arguments_fcout.txt");

		unordered_map<long, struct grptools::QualityScore> quantized;
		string quantized_to_string();
		string quantized_fcout(const string &fn = "quantized_fcout.txt");

		struct grptools::RecalTable0 recalTable0;
		string recalTable0_to_string();
		string recalTable0_fcout(const string &fn = "recalTable0_fcout.txt");

		struct grptools::RecalTable1 RecalTable1[globalB::GrpRT1LENGTH];
		string RecalTable1_to_string();
		struct grptools::RecalTable2 ***RecalTable2 = NULL; //[2][256][256]:covariateIdx, quality, covariateKey
		string RecalTable2_to_string();

		string grp_tables_fcout(const string &fn = "all_fcout.recal_data.grp.txt");
		void load(string &grp_path);
};






// ****** cpps, unordered ******
//#include "bed_record.cpp"
// bed_record__ start
bed_record::bed_record(const string& e_line)
{
	const int vec_sz = 3;
	vector<string> vec_line = bgitools::split_str_2_vec(e_line, '\t', vec_sz);
	chrname = vec_line[0];
	start = (uint32_t)atoi(vec_line[1].c_str()) + 1; // bed intervals used: [start+1, end]
	end = (uint32_t)atoi(vec_line[2].c_str());
	//snp_name = vec_line[3];
	//snp_type = vec_line[4];
}
// bed_record__ end



//#include "bgitools.cpp"
// bgitools__ start
vector<string> bgitools::split_str_2_vec(const string & str, const char & delimiter, const int & retainNum, int flag_push_tail_str)
{
	std::vector<std::string>   vec_ret{};
	vec_ret.reserve(16);

	std::stringstream  data(str);

	std::string line;
	if (retainNum == -1)
	{
		if (delimiter == ' ')
		{
			while (std::getline(data, line, delimiter)) // assume
			{
				// Note: if multiple delimitor in the source string,
				//           you may see many empty item in vector
				if (line == "") continue;
				vec_ret.push_back(line);
			}
			return vec_ret;
		}
		while (std::getline(data, line, delimiter))     // assume
		{
			// Note: if multiple delimitor in the source string,
			//       you may see many empty item in vector
			vec_ret.push_back(line);
		}
	}
	else
	{
		if (delimiter == ' ')
		{
			int cnt = 0;
			while (std::getline(data, line, delimiter)) // assume
			{
				if (cnt >= retainNum) break;
				if (line == "") continue;
				vec_ret.push_back(line);
				cnt++;
			}
			return vec_ret;
		}
		int cnt = 0;
		while (cnt < retainNum && std::getline(data, line, delimiter))     // assume
		{
			//if (cnt >= retainNum) break;
			vec_ret.push_back(line);
			cnt++;
		}
	}

	if (flag_push_tail_str)
	{
		std::getline(data, line, '\a');
		vec_ret.push_back(line);
	}
	return vec_ret;
}

void bgitools::trim_str(char * str)
{
	int len = (int)strlen(str);
	if (str[len - 1] == '\r' || str[len - 1] == '\n')
	{
		str[len - 1] = 0;
	}
}

void bgitools::trim_str(string & str)
{
	int len = str.length();
	if (str[len - 1] == '\r' || str[len - 1] == '\n')
	{
		str = str.substr(0, len - 1);
	}
}

string bgitools::fcout(const string & fn, const string & fc, const std::ios::openmode & F_W)
{
	ofstream if_(fn.c_str(), F_W);
	if (!if_.is_open()) {
		cout << "- make sure the file path is accessible!" << endl;
		assert(if_.is_open());
	}

	if_ << fc;
	if_.close();

	return fn;
}

double bgitools::round_(double val, int precision)
{
	std::stringstream s;
	s << std::setprecision(precision) << std::setiosflags(std::ios_base::fixed) << val;
	s >> val;
	return val;
}

string bgitools::join_vec_2_str(vector<string>& vec_string, const char & delimiter)
{
	string sb("");
	auto sz = vec_string.size();
	assert(sz >= 2);

	for (size_t i = 0; i < sz - 1; i++)
	{
		sb += vec_string[i] + delimiter;
	}
	sb += vec_string[sz - 1];

	return sb;
}

int bgitools::run_cmd(const char * cmd, char * res)
{
	FILE *pf = NULL;
	pf = popen(cmd, "r");

	if (pf == NULL)
	{
		printf("Error opening file unexist.ent: %s\n", strerror(errno));
	}

	assert(NULL != pf);

	const int TO_READ_SZ = 1024 * 4 - 1; // res must be a stack var!!!
	if (0 != fread(res, TO_READ_SZ, 1, pf))
	{
		pclose(pf);
		return -1;
	};

	pclose(pf);
	return 0;
}

template<typename T> string bgitools::to_string(const T &t)
{
	ostringstream ss;
	ss << t;
	return ss.str();
}

	template<typename TK, typename TV>
std::vector<TK> bgitools::extract_keys(std::unordered_map<TK, TV> const& input_map)
{
	std::vector<TK> retval;
	for (auto const& element : input_map) {
		retval.push_back(element.first);
	}
	return retval;
}

vector<bed_record> bgitools::read_bed_file_2_vec_bed_record(const string &fn)
{
	vector<bed_record> vec_record = {};
	vec_record.reserve(1723 * 2);

	auto F_R = (ios::in);
	ifstream if_(fn.c_str(), F_R);  assert(if_.is_open());

	const int MAX_CHAR = 1024;
	char line_content[MAX_CHAR];

	while (!if_.eof())
	{
		if_.getline(line_content, MAX_CHAR);	// don't need read too long

		if (line_content[0] != 'c') { continue; } // should be "chrN"
		string e_line = string(line_content);

		auto e_record = bed_record(e_line);
		vec_record.push_back(e_record);
	}
	if_.close();
	return vec_record; // chrX => ( [66763774 66766704], [66862998 66863349], ... )
}

unordered_map<string, vector<bed_record> > bgitools::vec_bed_record_2_map(vector<bed_record> & vec_bed_record)
{
	unordered_map<string, vector<bed_record> >  ret_map = {};
	for (auto & e_record : vec_bed_record)
	{
		if (0 == ret_map[e_record.chrname].size())
		{
			ret_map[e_record.chrname] = vector<bed_record>{};
		}
		ret_map[e_record.chrname].push_back(e_record);
	}
	return ret_map;
}
bool bgitools::bed_filter_v1(sam_record& e_sam_record, vector<bed_record> & sorted_vec_interval)
{ //binary_search_version

	// if false, we didn't mark it as known SNP, then we go on next step...of bed filter
	int alignmentStart = e_sam_record.pos;
	int alignmentEnd = getAlignmentEnd(e_sam_record, false);

	size_t mid, left = 0;
	size_t right = sorted_vec_interval.size();

	while (left < right)
	{
		//mid = left + (right - left) / 2;
		mid = (right + left) / 2;
		if (alignmentStart > sorted_vec_interval[mid].end)
		{
			left = mid + 1;
		}
		else if (alignmentEnd < sorted_vec_interval[mid].start)
		{
			right = mid;
		}
		else
		{
			return false;
		}
	}
	return true;
}

unordered_map<int, struct bgitools::CIGAR> bgitools::get_cigar(sam_record& e_sam_record)
{
	if (e_sam_record.cigar_map.size() == 0)
	{
		e_sam_record.cigar_map = bgitools::decode_cigar(e_sam_record.cigar);
		return e_sam_record.cigar_map;
	}

	return e_sam_record.cigar_map;
}

unordered_map<int, struct bgitools::CIGAR> bgitools::decode_cigar(string & cigar)
{

	auto cigar_char_to_enum_val = [](char & c)
	{
		OPT ret_opt = OPT::M;

		switch (c)
		{
			// total 9 CHAR
			case '=':
				ret_opt = OPT::EQ; break;
			case 'N':
				ret_opt = OPT::N; break;
			case 'H':
				ret_opt = OPT::H; break;
			case 'P':
				ret_opt = OPT::P; break;

			case 'D':
				ret_opt = OPT::D; break;
			case 'I':
				ret_opt = OPT::I; break;
			case 'S':
				ret_opt = OPT::S; break;
			case 'M':
				ret_opt = OPT::M; break;

			case 'X':
				ret_opt = OPT::X; break;
			default:
				cout << "error?" << endl;
				assert(c == "- not a valid OPT"[0]); break;
		}
		return ret_opt;
	};  // end fun() define


	unordered_map<int, struct bgitools::CIGAR>  cigar_map = {};
	assert(cigar.length() != 0);

	int prev = 0, last = 0;
	int offset = 0;
	for (int i = 0; i < cigar.length(); i++)
	{
		if (isdigit(cigar.c_str()[i]))
		{
			last += 1;
		}
		else
		{
			cigar_map[offset].opt = cigar_char_to_enum_val(cigar[i]);
			cigar_map[offset].length = atoi(cigar.substr(prev, last).c_str());

			prev = i + 1;
			last = 0;
			offset += 1;
		}
	}

	return cigar_map;
}

bool bgitools::sam_filter(sam_record& e_sam_record)
{
	const static uint8_t MAPPING_QUALITY_UNAVAILABLE = 255;
	const static uint32_t NO_ALIGNMENT_START = 0;
	const static uint16_t NOT_PRIMARY_ALIGNMENT_FLAG = 0x100;
	const static uint16_t DUPLICATE_READ_FLAG = 0x400;
	const static uint16_t READ_FAILS_VENDOR_QUALITY_CHECK_FLAG = 0x200;

	//    bool MappingQualityUnavailableFilter = (e_sam_record.mapq == MAPPING_QUALITY_UNAVAILABLE);
	//    bool MappingQualityZeroFilter = (e_sam_record.mapq == 0);

	//    bool isReadUnmapped = samtools::getReadUnmappedFlag(e_sam_record);
	//    bool isReadAlignment = (e_sam_record.pos == NO_ALIGNMENT_START);
	//    bool UnmappedReadFilter = (isReadUnmapped || isReadAlignment);

	//    bool NotPrimaryAlignmentFilter = ((e_sam_record.flag & NOT_PRIMARY_ALIGNMENT_FLAG) != 0);
	//    bool DuplicateReadFilter = ((e_sam_record.flag & DUPLICATE_READ_FLAG) != 0);
	//    bool FailsVendorQualityCheckFilter = ((e_sam_record.flag & READ_FAILS_VENDOR_QUALITY_CHECK_FLAG) != 0);
	return ((e_sam_record.mapq == MAPPING_QUALITY_UNAVAILABLE) ||
			(e_sam_record.mapq == 0) ||
			samtools::getReadUnmappedFlag(e_sam_record) ||
			(e_sam_record.pos == NO_ALIGNMENT_START) ||
			((e_sam_record.flag & NOT_PRIMARY_ALIGNMENT_FLAG) != 0) ||
			((e_sam_record.flag & DUPLICATE_READ_FLAG) != 0) ||
			((e_sam_record.flag & READ_FAILS_VENDOR_QUALITY_CHECK_FLAG) != 0) ||
			bgitools::MalformedReadFilter(e_sam_record));
}


struct bgitools::CigarOperator bgitools::cigarOperator(bgitools::OPT & opt)
{
	struct bgitools::CigarOperator co;
	switch (opt) {
		case bgitools::OPT::N:
			co.consumesReadBases = false;
			co.consumesReferenceBases = true;
			co.character = 'N';
			break;
		case OPT::H:
			co.consumesReadBases = false;
			co.consumesReferenceBases = false;
			co.character = 'H';
			break;
		case OPT::P:
			co.consumesReadBases = false;
			co.consumesReferenceBases = false;
			co.character = 'P';
			break;
		case OPT::D:
			co.consumesReadBases = false;
			co.consumesReferenceBases = true;
			co.character = 'D';
			break;
		case OPT::I:
			co.consumesReadBases = true;
			co.consumesReferenceBases = false;
			co.character = 'I';
			break;
		case OPT::S:
			co.consumesReadBases = true;
			co.consumesReferenceBases = false;
			co.character = 'S';
			break;
		case OPT::M:
			co.consumesReadBases = true;
			co.consumesReferenceBases = true;
			co.character = 'M';
			break;
		case OPT::EQ:
			co.consumesReadBases = true;
			co.consumesReferenceBases = true;
			co.character = '=';
			break;
		case OPT::X:
			co.consumesReadBases = true;
			co.consumesReferenceBases = true;
			co.character = 'X';
			break;
		default:
			printf("error");
	}

	return co;
}

bool bgitools::MalformedReadFilter(sam_record & e_sam_record)
{
	const static uint32_t NO_ALIGNMENT_START = 0;
	bool filterMismatchingBaseAndQuals = false;

	/**
	 * flag & READ_UNMAPPED_FLAG != 0 && (e_sam_record.pos == -1 || e_sam_record.pos == 0)
	 */
	auto checkInvalidAlignmentStart = [](sam_record& e_sam_record)
	{
		// GATK: read is not flagged as 'unmapped', but alignment start is NO_ALIGNMENT_START
		if (!samtools::getReadUnmappedFlag(e_sam_record) && (e_sam_record.pos == NO_ALIGNMENT_START))
		{
			return false;
		}
		// GATK: Read is not flagged as 'unmapped', but alignment start is -1
		if (!samtools::getReadUnmappedFlag(e_sam_record) && (e_sam_record.pos == -1))
		{
			return false;
		}

		return true;
	};

	/**
	 * flag & READ_UNMAPPED_FLAG != 0 && bgitools::getAlignmentEnd(e_sam_record, false) != -1) &&
	 * (bgitools::getAlignmentEnd(e_sam_record, false) - e_sam_record.pos + 1 < 0)
	 */
	auto checkInvalidAlignmentEnd = [](sam_record& e_sam_record)
	{
		if (!samtools::getReadUnmappedFlag(e_sam_record) && (bgitools::getAlignmentEnd(e_sam_record, false) != -1) &&
				(bgitools::getAlignmentEnd(e_sam_record, false) - e_sam_record.pos + 1 < 0))
		{
			return false;
		}
		return true;
	};


	//TODO: add RG check condition
	/**
	 * e_sam_record.seq.size() == e_sam_record.qual.size()
	 */
	auto checkMismatchingBasesAndQuals = [](sam_record& e_sam_record, bool& filterMismatchingBaseAndQuals)
	{
		bool result;
		if (e_sam_record.seq.size() == e_sam_record.qual.size())
		{
			result = true;
		}
		else if (filterMismatchingBaseAndQuals)
		{
			result = false;
		}
		else
		{
			printf("BAM file has a read with mismatching number of bases and base qualities");
			assert(0 == 1);
		}

		return result;
	};

	//    auto checkCigarDisagreesWithAlignment = [](sam_record& e_sam_record)
	//    {
	//        // Read has a valid alignment start, but the CIGAR string is empty
	//        if (!samtools::getReadUnmappedFlag(e_sam_record) &&
	//            (e_sam_record.pos != -1) &&
	//            (e_sam_record.pos != NO_ALIGNMENT_START)
	//                )
	//        {
	//            //TODO: when add read.getAlignmentBlocks().size() < 0, return false
	//            return true;
	//        }
	//        return true;
	//    };

	return (!checkInvalidAlignmentStart(e_sam_record) || !checkInvalidAlignmentEnd(e_sam_record) ||
			!checkMismatchingBasesAndQuals(e_sam_record, filterMismatchingBaseAndQuals));// ||
	//            !checkCigarDisagreesWithAlignment(e_sam_record));
}

int bgitools::getAlignmentEnd(sam_record & e_sam_record, const bool & cacheCigar)
{
	unordered_map<int, struct bgitools::CIGAR> cigar_map;
	if (cacheCigar)
	{
		cigar_map = bgitools::get_cigar(e_sam_record);
	}
	else
	{
		cigar_map = bgitools::decode_cigar(e_sam_record.cigar);
	}
	int length = 0;
	for (int iii = 0; iii < cigar_map.size(); iii++)
	{
		struct bgitools::CIGAR elt = cigar_map[iii];
		switch (elt.opt)
		{
			case bgitools::OPT::M:
			case bgitools::OPT::D:
			case bgitools::OPT::N:
			case bgitools::OPT::EQ:
			case bgitools::OPT::X:
				length += elt.length;
				break;
			default: {}

		}
	}

	return length + e_sam_record.pos - 1;
}

string bgitools::safeGetRefSeq(sam_record & e_sam_record, faidx_t *fai)
{
	string rname = e_sam_record.rname;
	uint32_t readStart = e_sam_record.pos;
	uint32_t readEnd = bgitools::getAlignmentEnd(e_sam_record);

	uint32_t offset = globalB::cb / 2; //ensure offset = 3!!!
	assert(offset == 3);

	uint32_t start = max(int(readStart - offset - samtools::getInsertionOffset(e_sam_record, 0)), 0);
	// GATK: long stop = (includeClippedBases ? read.getUnclippedEnd() :
	uint32_t stop = readEnd + offset + samtools::getInsertionOffset(e_sam_record, -1);

	string chr_start_end = rname + ":" + bgitools::to_string(start) + "-" + bgitools::to_string(stop);
	int seq_len;
	char * seq = fai_fetch(fai, chr_start_end.c_str(), &seq_len);
	if (seq_len < 0)
	{
		fprintf(stderr, "Failed to fetch sequence in %s\n", chr_start_end.c_str());
	}
	string ref = seq;
	free(seq);

	return ref;
}

string bgitools::safeGetRefSeq_v1(sam_record & e_sam_record, string & chromosome)
{
	string rname = e_sam_record.rname;
	uint32_t readStart = e_sam_record.pos;
	uint32_t readEnd = bgitools::getAlignmentEnd(e_sam_record);

	uint32_t offset = globalB::cb / 2; //ensure offset = 3!!!
	assert(offset == 3);

	uint32_t start = max(int(readStart - offset - samtools::getInsertionOffset(e_sam_record, 0)), 0);
	// GATK: long stop = (includeClippedBases ? read.getUnclippedEnd() :
	uint32_t stop = readEnd + offset + samtools::getInsertionOffset(e_sam_record, -1);
	if (start - 1 < 0)
	{
		cout << "sam_record pos:" << e_sam_record.pos << endl;
		assert(start - 1 >= 0);
	}
	//TODO: if bed file has start = 0, then GATK will take last line character which is "\n" in hg19.fasta
	//TODO: as one of the bases in ref sequence. So we should some day keep the same code as GATK on how to get ref sequence.
	string ref = chromosome.substr(start - 1, stop - start + 1);
	return ref;
}


bool bgitools::isRegularBase(const char & base)
{
	bool A = (base == 'A' || (base == 'a') || (base == '*'));// GATK: the wildcard character counts as an A
	bool C = ((base == 'C') || (base == 'c'));
	bool G = ((base == 'G') || (base == 'g'));
	bool T = ((base == 'T') || (base == 't'));

	if (A || C || G || T)
	{
		return true;
	}
	return false;
}

bool bgitools::isLowQualityBase(sam_record& e_sam_record, const int& offset)
{
	int baseQual = e_sam_record.qual[offset] - 33;
	if (baseQual < globalB::MIN_USABLE_Q_SCORE)
	{
		return true;
	}
	return false;
}

vector<pair<int, int>> bgitools::getFeaturesByHtslib_v2(string & chr_start_end, string & fn_vcf, int & n_vcf)
{
	const char * fname;

	vector<pair<int, int>> features;
	string vcf_lines = "";
	kstring_t str = { 0,0,0 };

	fname = fn_vcf.c_str();

	htsFile *fp = hts_open(fname, "r");
	if (!fp) printf("Could not read %s\n", fname);

	hts_itr_t *itr = tbx_itr_querys(globalB::vec_vcfPointer[n_vcf].tbx, chr_start_end.c_str()); //regs[i] = chr_start_end
	assert(itr); //htslib:if ( !itr ) continue;
	while (tbx_itr_next(fp, globalB::vec_vcfPointer[n_vcf].tbx, itr, &str) >= 0)
	{
		char str_start[512];
		int len_pos = 0;
		int len_ref = 0;
		int ii = 0;
		int meetDelimiterTimes = 0;
		while (true)
		{
			if (str.s[ii] == '\t')
			{
				meetDelimiterTimes++;
				ii++;
			}
			if (meetDelimiterTimes == 1)
			{
				const int pos_start_ind = ii;
				while (str.s[ii] != '\t')
				{
					len_pos++;
					ii++;
				}
				strncpy(str_start, str.s + pos_start_ind, len_pos);
				meetDelimiterTimes++;
				ii++;
				continue;
			}
			if (meetDelimiterTimes == 3)
			{
				while (str.s[ii] != '\t')
				{
					len_ref++;
					ii++;
				}
				break;
			}
			ii++;
		}
		int indel_start;
		sscanf(str_start, "%d", &indel_start);
		int indel_end = indel_start + len_ref - 1;
		features.push_back(pair<int, int>{indel_start, indel_end});
	}
	if (hts_close(fp)) printf("hts_close returned non-zero status: %s\n", fname);
	free(str.s);
	free(itr);

	return features;
}

map<int, vector<pair<int, int>>*> bgitools::getMapFeatures(string & rname, int & start, int & end, vector<string> & vec_fn_vcf)
{
	string chr_start_end = rname + ":" + bgitools::to_string(start) + "-" + bgitools::to_string(end);

	map<int, vector<pair<int, int>>*> map_features;

	for (int n_vcf = 0; n_vcf < vec_fn_vcf.size(); n_vcf++)
	{
		vector<pair<int, int>> features = bgitools::getFeaturesByHtslib_v2(chr_start_end, vec_fn_vcf[n_vcf], n_vcf);
		map_features[n_vcf] = &features;
	}

	return map_features;
}

int bgitools::readInt(char * &in)
{
	unsigned int byte1 = (unsigned)((*in) << 24); in++;
	unsigned int byte2 = (unsigned)((*in) << 24); in++;
	unsigned int byte3 = (unsigned)((*in) << 24); in++;
	int byte4 = (*in); in++;

	return (byte4 << 24)
		+ (byte3 >> 8)
		+ (byte2 >> 16)
		+ (byte1 >> 24);
}

string bgitools::readString(char * &in)
{
	string bis("");
	char b;
	while ((b = (int)(*in)) != 0)
	{
		assert(b >= 0);
		bis += b;
		in++;
	}
	in++;
	return bis;
}

long bgitools::readLong(char * &in, char *& is, ifstream & _if, const int & maxSize)
{
	char buffer[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	int i = 0;
	while ((in - is) < maxSize && i < 8)
	{
		buffer[i] = (*in);
		i++;
		in++;
	}
	if (in - is >= maxSize)
	{
		memset(is, 0, sizeof(char)*maxSize);
		_if.readsome(is, maxSize);
		in = is;
		for (int ii = i; ii < 8; ii++)
		{
			buffer[ii] = (*in);
			in++;
		}
	}
	unsigned long byte1 = (unsigned long)(((long)buffer[0]) << 56);
	unsigned long byte2 = (unsigned long)(((long)buffer[1]) << 56);
	unsigned long byte3 = (unsigned long)(((long)buffer[2]) << 56);
	unsigned long byte4 = (unsigned long)(((long)buffer[3]) << 56);
	unsigned long byte5 = (unsigned long)(((long)buffer[4]) << 56);
	unsigned long byte6 = (unsigned long)(((long)buffer[5]) << 56);
	unsigned long byte7 = (unsigned long)(((long)buffer[6]) << 56);
	long byte8 = (long)buffer[7];
	return (byte8 << 56) + (byte7 >> 8) + (byte6 >> 16) + (byte5 >> 24) + (byte4 >> 32) + (byte3 >> 40) + (byte2 >> 48) + (byte1 >> 56);
}

void bgitools::readSequenceDictionary(char * &in)
{
	int size = bgitools::readInt(in);
	if (size < 0)
	{
		cout << "Size of the sequence dictionary entries is negative" << endl;
		assert(size >= 0);
	}
	for (int x = 0; x < size; x++)
	{
		bgitools::readString(in);
		bgitools::readInt(in);
	}
}

vector<pair<string, string>> bgitools::readHeader(char * &in, char *& is, ifstream & _if, const int & maxSize)
{
	int version = bgitools::readInt(in);
	if (version != 3)
	{
		cout << "not version 3, only support version 3 now!" << endl;
		assert(version == 3);
	}
	string indexedFile = bgitools::readString(in);
	long indexFileSize = bgitools::readLong(in, is, _if, maxSize);
	long indexedFileTS = bgitools::readLong(in, is, _if, maxSize);
	string indexedFileMD5 = bgitools::readString(in);

	int flags = bgitools::readInt(in);
	if (version < 3 && (flags & globalB::SEQUENCE_DICTIONARY_FLAG) == globalB::SEQUENCE_DICTIONARY_FLAG) {
		readSequenceDictionary(in);
	}

	vector<pair<string, string>> properties;
	if (version >= 3) {
		int nProperties = bgitools::readInt(in);
		while (nProperties-- > 0) {
			string key = bgitools::readString(in);
			string value = bgitools::readString(in);
			properties.push_back({ key, value });
		}
	}

	return properties;
}

void bgitools::ChrIndex::read(char * &in, char *& is, ifstream & _if, const int & maxSize)
{
	name = bgitools::readString(in);
	binWidth = bgitools::readInt(in);
	int nBins = bgitools::readInt(in);
	longestFeature = bgitools::readInt(in);
	//GATK:largestBlockSize = dis.readInt();
	// GATK:largestBlockSize and totalBlockSize are old V3 index values.  largest block size should be 0 for
	// all newer V3 block.  This is a nasty hack that should be removed when we go to V4 (XML!) indices
	OLD_V3_INDEX = bgitools::readInt(in) > 0;
	nFeatures = bgitools::readInt(in);

	// GATK:note the code below accounts for > 60% of the total time to read an index
	long pos = bgitools::readLong(in, is, _if, maxSize);

	for (int binNumber = 0; binNumber < nBins; binNumber++)
	{
		long nextPos = bgitools::readLong(in, is, _if, maxSize);
		long size = nextPos - pos;
		blocks.push_back(Block{ pos, size });
		pos = nextPos;
	}
}

bgitools::Block bgitools::ChrIndex::getBlocks(int & start, int & end)
{
	bgitools::Block block;
	if (blocks.size() == 0)
	{
		return block;
	}
	else
	{
		// GATK:Adjust position for the longest feature in this chromosome.  This insures we get
		// features that start before the bin but extend into it
		int adjustedPosition = max(start - longestFeature, 0);
		int startBinNumber = adjustedPosition / binWidth;
		if (startBinNumber >= blocks.size()) // are we off the end of the bin list, so return nothing
			return block;
		else {
			const int batchNum0 = (end - 1) / binWidth;
			const int batchNum1 = blocks.size() - 1;
			int endBinNumber = min(batchNum0, batchNum1);

			// By definition blocks are adjacent for the liner index.  Combine them into one merged block

			long startPos = blocks[startBinNumber].startPosition;
			long endPos = blocks[endBinNumber].startPosition + blocks[endBinNumber].size;
			long size = endPos - startPos;
			if (size == 0)
			{
				return block;
			}
			else {
				bgitools::Block mergedBlock;
				mergedBlock.startPosition = startPos;
				mergedBlock.size = size;
				return mergedBlock;
			}
		}
	}

}

string bgitools::getFeatureReader(string & inputFile)
{

}
#if 0
pair<vector<pair<string, string>>, unordered_map<string, bgitools::ChrIndex>> bgitools::getFeatureSource(string & inputFile)
{
	string indexFile = inputFile + globalB::STANDARD_INDEX_EXTENSION;
	auto F_R = (ios::binary);
	ifstream if_(indexFile.c_str(), F_R);  assert(if_.is_open());
	const int maxSize = 512000;
	char * is = (char*)malloc(sizeof(char)*maxSize);
	if_.readsome(is, sizeof(char)*maxSize);

	char * in = is;
	int magicNumber = bgitools::readInt(in);
	int type = bgitools::readInt(in);

	vector<pair<string, string>> properties = bgitools::readHeader(in, is, if_, maxSize);
	int nChromosomes = bgitools::readInt(in);

	unordered_map<string, bgitools::ChrIndex> chrIndices;
	while (nChromosomes-- > 0)
	{
		struct bgitools::ChrIndex chrIdx;
		chrIdx.read(in, is, if_, maxSize);
		chrIndices[chrIdx.name] = chrIdx;
		//        chrIndices.put(chrIdx.getName(), chrIdx);
	}
	if_.close();
	free(is);
	//    vector<pair<string, bgitools::SAMSequenceRecord>> sequenceDictionary = bgitools::getSequenceDictionaryFromProperties(idx.first);
	return pair<vector<pair<string, string>>, unordered_map<string, bgitools::ChrIndex>>{properties, chrIndices};
}
#endif 

#if 0
vector<unordered_map<string, bgitools::ChrIndex>> bgitools::loadVcfIdx(vector<string> & vec_fn_vcf)
{
	vector<unordered_map<string, bgitools::ChrIndex>> vec_vcf_idx;
	for (auto & e_fn_vcf : vec_fn_vcf)
	{
		pair<vector<pair<string, string>>, unordered_map<string, bgitools::ChrIndex>> featureSource = bgitools::getFeatureSource(e_fn_vcf);
		vec_vcf_idx.push_back(featureSource.second);
	}
	return vec_vcf_idx;
}
#endif 
#if 0
string bgitools::vcfReadLine(char * &fin, char *& fis, ifstream & _fif, const int & maxSize)
{
	char lineBuffer[10000] = { 0 }; //one line <= 10000 char?
	int linePosition = 0;
	//    string lineBuffer = "";

	long sizeRead;
	while (true)
	{
		int b = (*fin); fin++;
		if (fin - fis >= maxSize)
		{
			memset(fis, 0, sizeof(char)*maxSize);
			sizeRead = _fif.readsome(fis, sizeof(char)*maxSize);
			fin = fis;
			if (sizeRead <= 0)
			{
				// GATK:eof reached.  Return the last line, or null if this is a new line
				if (linePosition > 0)
				{
					return string(lineBuffer, lineBuffer + linePosition);
					//                return lineBuffer.substr(0, linePosition);
				}
				else
				{
					return "";
				}
			}
		}

		char c = (char)(b & 0xFF);
		if (c == globalB::LINEFEED || c == globalB::CARRIAGE_RETURN)
		{
			if (c == globalB::CARRIAGE_RETURN && (*(fin + 1)) == globalB::LINEFEED)
			{
				fin++; // <= skip the trailing \n in case of \r\n termination
				if (fin - fis >= maxSize)
				{
					memset(fis, 0, sizeof(char)*maxSize);
					_fif.readsome(fis, sizeof(char)*maxSize);
					fin = fis;
				}
			}

			return string(lineBuffer, lineBuffer + linePosition);
		}
		else
		{
			// GATK:Expand line buffer size if neccessary.  Reserve at least 2 characters
			// for potential line-terminators in return string

			//            if (linePosition > (lineBuffer.length() - 3)) {
			////                char * temp = new char[B::BUFFER_OVERFLOW_INCREASE_FACTOR * lineBuffer.length()];
			//                string temp = lineBuffer.substr(0, lineBuffer.length());
			//                lineBuffer = temp;
			//            }

			lineBuffer[linePosition++] = c;
		}
	}
	return lineBuffer;
}
#endif 

#if 0
string bgitools::vcfDecode(char * &fin, char *& fis, ifstream & _fif, const int & maxSize)
{
	string line = bgitools::vcfReadLine(fin, fis, _fif, maxSize);

	if (line[0] == globalB::HEADER_INDICATOR)
	{
		return "";
	}

	return line;
}
#endif 

#if 0
map<int, vector<pair<int, int>>> bgitools::getFeaturesByVcfIdx(string & rname, int & start, int & end, vector<string> & vec_fn_vcf)
{
	vector<unordered_map<string, bgitools::ChrIndex>> vec_vcf_idx;
	for (auto & e_fn_vcf : vec_fn_vcf)
	{
		pair<vector<pair<string, string>>, unordered_map<string, bgitools::ChrIndex>> featureSource = bgitools::getFeatureSource(e_fn_vcf);
		vec_vcf_idx.push_back(featureSource.second);
	}
	vec_vcf_idx = bgitools::loadVcfIdx(vec_fn_vcf);

	int n_vcf = 0;
	map<int, vector<pair<int, int>>> map_features;
	vector<pair<int, int>> features;
	char * fis = NULL;
	for (auto & e_idx : vec_vcf_idx)
	{
		int blockStart = start - 1;
		bgitools::Block blk = e_idx[rname].getBlocks(blockStart, end);

		if (blk.startPosition != -1)
		{
			string fn_vcf = vec_fn_vcf[n_vcf];  auto F_R = (ios::in);
			ifstream fif_(fn_vcf.c_str(), F_R);  assert(fif_.is_open());
			//            ifstream *fif_ = B::vec_ifs[n_vcf];

			fif_.seekg(blk.startPosition);
			const int size1 = blk.size > 100000000 ? 10000000 : blk.size;
			const int bufferSize = min(2000000, size1);
			//            if (fis == NULL)
			//            {
			fis = (char*)malloc(sizeof(char)*bufferSize);
			//            }
			//            else
			//            {
			//                void * new_fis = realloc(fis, sizeof(char)*bufferSize);
			//                if (!new_fis) assert(new_fis);
			//                fis = (char *)new_fis;
			//            }
			long sizeRead = fif_.readsome(fis, sizeof(char)*bufferSize);
			if (sizeRead <= 0)
			{
				cout << "exception in reading vcf contents" << endl;
				assert(sizeRead > 0);
			}

			char * fin = fis;
			while (true)
			{
				string lineTmp = bgitools::vcfDecode(fin, fis, fif_, bufferSize);
				if (lineTmp == "") continue;
				vector<string> vec_line = bgitools::split_str_2_vec(lineTmp, '\t', 4); //only formmer 4 column info is used.
				int featureStart = atoi(vec_line[1].c_str());
				if (featureStart > end)
				{
					break;
				}

				int featureEnd = featureStart + vec_line[3].length() - 1;
				if (featureEnd < start) continue;

				features.push_back({ featureStart, featureEnd });
			}
			free(fis);
		}
		map_features[n_vcf] = features;
		n_vcf++;
	}

	return map_features;
}
#endif 
#if 0
vector<pair<int, int>> bgitools::getVecFeaturesByVcfIdx(string & rname, int & start, int & end, string & fn_vcf, int & n_vcf)
{
	unordered_map<string, bgitools::ChrIndex> vcf_idx;
	pair<vector<pair<string, string>>, unordered_map<string, bgitools::ChrIndex>> featureSource = bgitools::getFeatureSource(fn_vcf);
	vcf_idx = featureSource.second;

	vector<pair<int, int>> features;
	char * fis = NULL;
	int blockStart = start - 1;
	bgitools::Block blk = vcf_idx[rname].getBlocks(blockStart, end);

	if (blk.startPosition != -1)
	{
		auto F_R = (ios::in);
		ifstream fif_(fn_vcf.c_str(), F_R);  assert(fif_.is_open());
		//            ifstream *fif_ = B::vec_ifs[n_vcf];

		fif_.seekg(blk.startPosition);
		const int size1 = blk.size > 100000000 ? 10000000 : blk.size;
		const int bufferSize = min(2000000, size1);
		//            if (fis == NULL)
		//            {
		fis = (char*)malloc(sizeof(char)*bufferSize);
		//            }
		//            else
		//            {
		//                void * new_fis = realloc(fis, sizeof(char)*bufferSize);
		//                if (!new_fis) assert(new_fis);
		//                fis = (char *)new_fis;
		//            }
		long sizeRead = fif_.readsome(fis, sizeof(char)*bufferSize);
		if (sizeRead <= 0)
		{
			cout << "exception in reading vcf contents" << endl;
			assert(sizeRead > 0);
		}

		char * fin = fis;
		while (true)
		{
			string lineTmp = bgitools::vcfDecode(fin, fis, fif_, bufferSize);
			if (lineTmp == "") continue;
			vector<string> vec_line = bgitools::split_str_2_vec(lineTmp, '\t', 4); //only formmer 4 column info is used.
			int featureStart = atoi(vec_line[1].c_str());
			if (featureStart > end)
			{
				break;
			}

			int featureEnd = featureStart + vec_line[3].length() - 1;
			if (featureEnd < start) continue;

			features.push_back({ featureStart, featureEnd });
		}
		free(fis);
	}

	return features;
}
#endif 

bool bgitools::basesAreEqual(char & base1, char & base2)
{
	return samtools::simpleBaseToBaseIndex(base1) == samtools::simpleBaseToBaseIndex(base2);
}


// bgitools__ end



//#include "samtools.cpp"
// samtools__ start
pair<int, int> samtools::calculateQueryRange(sam_record& e_sam_record)
{
	pair<int, int> queryRange;
	int queryStart = -1, queryStop = -1;
	int readI = 0;

	unordered_map<int, struct bgitools::CIGAR> cigar_map = bgitools::get_cigar(e_sam_record);
	for (int iii = 0; iii < cigar_map.size(); iii++)
	{
		bgitools::OPT opt = cigar_map[iii].opt;
		switch (opt)
		{
			case bgitools::OPT::N:
				return make_pair(int(NULL), int(NULL)); //GATK: cannot handle these
			case bgitools::OPT::H:
			case bgitools::OPT::P:
			case bgitools::OPT::D:
				break; //GATK: ignore pads, hard clips, and deletions
			case bgitools::OPT::I:
			case bgitools::OPT::S:
			case bgitools::OPT::M:
			case bgitools::OPT::EQ:
			case bgitools::OPT::X:
				{
					int prev = readI;
					readI += cigar_map[iii].length;
					if (opt != bgitools::OPT::S)
					{
						if (queryStart == -1)
							queryStart = prev;
						queryStop = readI;
					}
					// GATK: in the else case we aren't including soft clipped bases, so we don't update
					// queryStart or queryStop
					break;
				}
			default:
				//                printf("BUG: Unexpected CIGAR element in read " + e_sam_record.qname);
				assert(0 == 1);
		}
	}

	if (queryStop == queryStart) {
		// GATK: this read is completely clipped away, and yet is present in the file for some reason
		return make_pair(int(NULL), int(NULL));
	}

	queryRange.first = queryStart;
	queryRange.second = queryStop;

	return queryRange;
}

uint32_t samtools::getInsertionOffset(sam_record& e_sam_record, const int & ind)
{
	//ind=0: first element; ind=-1: last element
	if (ind == 0)
	{
		unordered_map<int, struct bgitools::CIGAR> cigar_map = bgitools::get_cigar(e_sam_record);
		if (cigar_map[ind].opt == bgitools::OPT::I)
		{
			return cigar_map[ind].length;
		}
		else
		{
			return 0;
		}
	}
	else
	{   //firstOrLast = "last"
		unordered_map<int, struct bgitools::CIGAR> cigar_map = bgitools::get_cigar(e_sam_record);
		if (cigar_map[cigar_map.size() - 1].opt == bgitools::OPT::I)
		{
			return cigar_map[cigar_map.size() - 1].length;
		}
		else
		{
			return 0;
		}
	}
}


map<string, vector<sam_record*>> samtools::read_sam_file_2_map_vec_p_sam_record(const string & fn)
{
	sam_record::header_str = string("");
	sam_record::group_name = string("");
	sam_record::BI = string("");
	sam_record::BD = string("");

	map<string, vector<sam_record*>> map_chrname_vec{};

	const int MAX_CHAR = 1024;
	auto F_R = (ios::in);
	ifstream if_(fn.c_str(), F_R);  assert(if_.is_open());

	sam_record * p_e_sam_record = NULL;
	char line_content[MAX_CHAR];
	line_content[MAX_CHAR - 1] = '\0';
	if_.seekg(0, ios::beg);
	while (!if_.eof())
	{
		if_.getline(line_content, MAX_CHAR);	// don't need read too long

		// skip header content
		if (line_content[0] == '@')
		{
			sam_record::header_str += string(line_content) + "\n";
			continue;
		}
		if (line_content[0] != 'C') { continue; }

		string e_line = string(line_content);

		p_e_sam_record = new sam_record(e_line);

		if (bgitools::MalformedReadFilter(*p_e_sam_record))
		{
			continue;
		}

		auto sz_chr_old = map_chrname_vec.size();
		map_chrname_vec[p_e_sam_record->rname].push_back(p_e_sam_record);
		auto sz_chr_new = map_chrname_vec.size();

		if (sz_chr_old != sz_chr_new)
		{
			globalB::vec_chr.push_back(p_e_sam_record->rname);
		}

	}
	if_.close();


	// from header to get  group_name
	auto *target_sm = "\tSM:";
	auto len_target = strlen(target_sm);
	auto loc_sm = sam_record::header_str.find(target_sm);
	assert(loc_sm != string::npos);
	auto loc_end = sam_record::header_str.find('@', loc_sm);
	sam_record::group_name = sam_record::header_str.substr(loc_sm + len_target, loc_end - loc_sm - len_target - 1);

	// get BD BI, static var
	assert(NULL != p_e_sam_record);
	auto len_qual = p_e_sam_record->qual.size();
	sam_record::BI = "BI:Z:" + string(len_qual, 'N');
	sam_record::BD = "BD:Z:" + string(len_qual, 'N');

	return map_chrname_vec;
}

void samtools::del_map_vec_p_sam_record(map<string, vector<sam_record *> >& map_chr_vec_p_sam_record)
{
	for (auto &e_m : map_chr_vec_p_sam_record)
	{
		for (auto &p_e_v : e_m.second)
		{
			delete p_e_v;
			p_e_v = nullptr;
		}

	}
	map_chr_vec_p_sam_record.clear();

}


bool samtools::getReadNegativeStrandFlag(sam_record& e_sam_record)
{
	return (e_sam_record.flag & globalB::READ_STRAND_FLAG) != 0;
}

string samtools::simpleReverseComplement(string & bases)
{
	string rcbases = "";

	int readLength = bases.length();
	for (int i = 0; i < readLength; i++)
	{
		switch (bases[readLength - 1 - i])
		{
			case 'A':
			case 'a':
				rcbases += 'T';
				break;
			case 'C':
			case 'c':
				rcbases += 'G';
				break;
			case 'G':
			case 'g':
				rcbases += 'C';
				break;
			case 'T':
			case 't':
				rcbases += 'A';
				break;
			default:
				rcbases += bases[readLength - 1 - i];
		}
	}

	return rcbases;
}

bool samtools::getReadPairedFlag(const int & flag)
{
	return (flag & globalB::READ_PAIRED_FLAG) != 0;
}

bool samtools::getSecondOfPairFlag(const int & flag)
{
	if (!samtools::getReadPairedFlag(flag))
	{
		printf("I");
		assert(0 == 1);
	}
	return (flag & globalB::SECOND_OF_PAIR_FLAG) != 0;
}

int samtools::keyFromCycle(int & cycle)
{
	// GATK:no negative values because values must fit into the first few bits of the long
	int result = abs(cycle);
	result = result << 1; // shift so we can add the "sign" bit
	if (cycle < 0)
		result++; // negative cycles get the lower-most bit set
	return result;
}

int samtools::simpleBaseToBaseIndex(const char & base)
{
	if (base < 0 || base >= 256)
	{
		auto a = base;
		cout << a << endl;
		printf("Non-standard bases were encountered in either the input reference or BAM file(s)");
		printf("Error opening file unexist.ent: %s\n", strerror(errno));
		assert(0 <= base && base < 256);
	}
	if (base == 'A' || base == 'a' || base == '*')
	{
		return 0;
	}
	if (base == 'C' || base == 'c')
	{
		return 1;
	}
	if (base == 'G' || base == 'g')
	{
		return 2;
	}
	if (base == 'T' || base == 't')
	{
		return 3;
	}
	return -1;
}

string samtools::baseIndexToSimpleBase(int & baseIndex)
{
	switch (baseIndex)
	{
		case 0:
			return "A";
		case 1:
			return "C";
		case 2:
			return "G";
		case 3:
			return "T";
		default:
			return ".";
	}
}

int samtools::keyFromContext(const string & bases, const int & start, const int & end)
{
	int key = end - start;
	int bitOffset = globalB::LENGTH_BITS;
	for (int i = start; i < end; i++) {
		int baseIndex = samtools::simpleBaseToBaseIndex(bases[i]);
		if (baseIndex == -1) // ignore non-ACGT bases
			return -1;
		key |= (baseIndex << bitOffset);
		bitOffset += 2;
	}
	return key;
}

sam_record samtools::clipLowQualEnds(sam_record & e_sam_record, const char & lowQual, const samtools::ClippingRepresentation & algorithm)
{
	string quals = e_sam_record.qual;

	int readLength = e_sam_record.seq.length();
	int leftClipIndex = 0;
	int rightClipIndex = readLength - 1;

	// check how far we can clip both sides
	while (rightClipIndex >= 0 && (quals[rightClipIndex] - 33) <= lowQual) rightClipIndex--;
	while (leftClipIndex < readLength && (quals[leftClipIndex] - 33) <= lowQual) leftClipIndex++;

	// if the entire read should be clipped, then return an empty read.
	if (leftClipIndex > rightClipIndex)
	{
		sam_record emptyRead = e_sam_record;
		emptyRead.pos = 0;
		emptyRead.seq = "";
		emptyRead.cigar = "";
		return emptyRead;
	}

	vector<pair<int, int>> op = {};
	if (rightClipIndex < readLength - 1)
	{
		pair<int, int> op1;
		op1.first = rightClipIndex + 1;
		op1.second = readLength - 1;
		op.push_back(op1);
	}
	if (leftClipIndex > 0)
	{
		pair<int, int> op2;
		op2.first = 0;
		op2.second = leftClipIndex - 1;
		op.push_back(op2);
	}

	if (op.size() == 0)
	{
		return e_sam_record;
	}

	for (pair<int, int> & e_op : op)
	{
		switch (algorithm)
		{
			case samtools::ClippingRepresentation::WRITE_NS:
				for (int i = 0; i < readLength; i++)
				{
					if (i >= e_op.first && i <= e_op.second) {
						e_sam_record.seq[i] = 'N';
					}
				}
				break;
			default:
				printf("Unexpected Clipping operator type");
				assert(0 == 1);
		}
	}

	return e_sam_record;
}

/**
 * Finds the adaptor boundary around the read and returns the first base inside the adaptor that is closest to
 * the read boundary. If the read is in the positive strand, this is the first base after the end of the
 * fragment (Picard calls it 'insert'), if the read is in the negative strand, this is the first base before the
 * beginning of the fragment.
 *
 * There are two cases we need to treat here:
 *
 * 1) Our read is in the reverse strand :
 *
 *     <----------------------| *
 *   |--------------------->
 *
 *   in these cases, the adaptor boundary is at the mate start (minus one)
 *
 * 2) Our read is in the forward strand :
 *
 *   |---------------------->   *
 *     <----------------------|
 *
 *   in these cases the adaptor boundary is at the start of the read plus the inferred insert size
 */
int samtools::getAdaptorBoundary(sam_record & e_sam_record)
{
	int MAXIMUM_ADAPTOR_LENGTH = 8;
	int insertSize = abs(e_sam_record.tlen);    // GATK:the inferred insert size can be negative if the mate is mapped before the read (so we take the absolute value)

	if (insertSize == 0 || ((e_sam_record.flag & globalB::READ_UNMAPPED_FLAG) != 0))                // GATK:no adaptors in reads with mates in another chromosome or unmapped pairs
	{
		return -1;
	}

	int adaptorBoundary;                                          // GATK:the reference coordinate for the adaptor boundary (effectively the first base IN the adaptor, closest to the read)
	if (0 != (e_sam_record.flag & globalB::READ_STRAND_FLAG))
	{ //read reverse strand
		adaptorBoundary = e_sam_record.pnext - 1;           // GATK:case 1 (see header)
	}
	else
	{ //mate reverse strand
		adaptorBoundary = e_sam_record.pos + insertSize + 1;  // GATK:case 2 (see header)
		// TODO: in GATK4, this will be fixed to: adaptorBoundary = e_sam_record.pos + insertSize;
	}

	if ((adaptorBoundary < e_sam_record.pos - MAXIMUM_ADAPTOR_LENGTH) || (adaptorBoundary > bgitools::getAlignmentEnd(e_sam_record) + MAXIMUM_ADAPTOR_LENGTH))
		adaptorBoundary = -1;                                       // GATK:we are being conservative by not allowing the adaptor boundary to go beyond what we belive is the maximum size of an adaptor

	return adaptorBoundary;
}


pair<bool, struct bgitools::CIGAR> samtools::readStartsWithInsertion(unordered_map<int, struct bgitools::CIGAR> & cigar_map)
{
	for (int i = 0; i < cigar_map.size(); i++)
	{
		struct bgitools::CIGAR elt = cigar_map[i];
		if (elt.opt == bgitools::OPT::I)
		{
			return pair<bool, struct bgitools::CIGAR>(true, elt);
		}
		else if (elt.opt != bgitools::OPT::H && elt.opt != bgitools::OPT::S)
			break;
	}

	struct bgitools::CIGAR cigar;
	cigar.length = -1;
	return pair<bool, struct bgitools::CIGAR>(false, cigar); // NULL cigar
}

int samtools::getReadLengthForCigar(unordered_map<int, struct bgitools::CIGAR> & cigar_map)
{
	int length = 0;
	for (int i = 0; i < cigar_map.size(); i++)
	{
		struct bgitools::CIGAR elt = cigar_map[i];
		if (bgitools::cigarOperator(elt.opt).consumesReadBases)
		{
			length += elt.length;
		}
	}
	return length;
}

int samtools::getSoftStart(unordered_map<int, struct bgitools::CIGAR> & cigar_map, uint32_t & alignmentStart)
{
	int softStart = alignmentStart;
	for (int i = 0; i < cigar_map.size(); i++)
	{
		struct bgitools::CIGAR elt = cigar_map[i];
		if (elt.opt == bgitools::OPT::S)
		{
			softStart -= elt.length;
		}
		else if (elt.opt != bgitools::OPT::H)
			break;
	}
	return softStart;
}

int samtools::getSoftEnd(unordered_map<int, struct bgitools::CIGAR> & cigar_map, uint32_t & alignmentEnd)
{
	bool foundAlignedBase = false;
	int softEnd = alignmentEnd;
	for (int i = cigar_map.size() - 1; i >= 0; --i)
	{
		struct bgitools::CIGAR elt = cigar_map[i];

		if (elt.opt == bgitools::OPT::S) // GATK:assumes the soft clip that we found is at the end of the aligned read
			softEnd += elt.length;
		else if (elt.opt != bgitools::OPT::S) {
			foundAlignedBase = true;
			break;
		}
	}
	if (!foundAlignedBase) { // GATK:for example 64H14S, the soft end is actually the same as the alignment end
		softEnd = alignmentEnd;
	}
	return softEnd;
}

int samtools::calculateHardClippingAlignmentShift(struct bgitools::CIGAR & cigarElement, const uint32_t & clippedLength)
{
	// GATK:Insertions should be discounted from the total hard clip count
	if (cigarElement.opt == bgitools::OPT::I)
		return -((int)clippedLength);

	// GATK:Deletions should be added to the total hard clip count
	else if (cigarElement.opt == bgitools::OPT::D)
		return cigarElement.length;

	// GATK:There is no shift if we are not clipping an indel
	return 0;
}

int samtools::calculateAlignmentStartShift(unordered_map<int, struct bgitools::CIGAR> & old_cigar_map, unordered_map<int, struct bgitools::CIGAR> & new_cigar_map)
{
	int newShift = 0;
	int oldShift = 0;

	bool readHasStarted = false;  // if the new cigar is composed of S and H only, we have to traverse the entire old cigar to calculate the shift
	for (int i = 0; i < new_cigar_map.size(); i++)
	{
		struct bgitools::CIGAR new_elt = new_cigar_map[i];
		if (new_elt.opt == bgitools::OPT::H || new_elt.opt == bgitools::OPT::S)
		{
			newShift += new_elt.length;
		}
		else
		{
			readHasStarted = true;
			break;
		}
	}

	for (int i = 0; i < old_cigar_map.size(); i++)
	{
		struct bgitools::CIGAR old_elt = old_cigar_map[i];
		if (old_elt.opt == bgitools::OPT::H || old_elt.opt == bgitools::OPT::S)
		{
			oldShift += old_elt.length;
		}
		else if (readHasStarted)
			break;
	}
	return newShift - oldShift;
}

struct samtools::CigarShift samtools::cleanHardClippedCigar(unordered_map<int, struct bgitools::CIGAR> & cigar_map)
{
	unordered_map<int, struct bgitools::CIGAR> clean_cigar_map;
	int cigar_ind = 0;
	int shiftFromStart = 0;
	int shiftFromEnd = 0;
	stack<struct bgitools::CIGAR> cigarStack;
	stack<struct bgitools::CIGAR> inverseCigarStack;

	for (int i = 0; i < cigar_map.size(); i++)
	{
		cigarStack.push(cigar_map[i]);
	}

	for (int i = 1; i <= 2; i++)
	{
		int shift = 0;
		int totalHardClip = 0;
		bool readHasStarted = false;
		bool addedHardClips = false;

		while (!cigarStack.empty())
		{
			struct bgitools::CIGAR cigarElement = cigarStack.top();
			cigarStack.pop();

			if (!readHasStarted &&
					//                        cigarElement.getOperator() != CigarOperator.INSERTION &&
					cigarElement.opt != bgitools::OPT::D &&
					cigarElement.opt != bgitools::OPT::H)
				readHasStarted = true;

			else if (!readHasStarted && cigarElement.opt == bgitools::OPT::H)
				totalHardClip += cigarElement.length;

			else if (!readHasStarted && cigarElement.opt == bgitools::OPT::D)
				totalHardClip += cigarElement.length;

			if (readHasStarted) {
				if (i == 1) {
					if (!addedHardClips) {
						if (totalHardClip > 0)
						{
							inverseCigarStack.push(bgitools::CIGAR{ totalHardClip, bgitools::OPT::H });
						}
						addedHardClips = true;
					}
					inverseCigarStack.push(cigarElement);
				}
				else {
					if (!addedHardClips) {
						if (totalHardClip > 0)
						{
							clean_cigar_map[cigar_ind] = bgitools::CIGAR{ totalHardClip, bgitools::OPT::H };
							cigar_ind++;
						}
						addedHardClips = true;
					}
					clean_cigar_map[cigar_ind] = bgitools::CIGAR{ cigarElement.length, cigarElement.opt };
					cigar_ind++;
				}
			}
		}
		// first pass  (i=1) is from end to start of the cigar elements
		if (i == 1)
		{
			shiftFromEnd = shift;
			cigarStack = inverseCigarStack;
		}
		// second pass (i=2) is from start to end with the end already cleaned
		else {
			shiftFromStart = shift;
		}
	}
	return samtools::CigarShift{ clean_cigar_map, shiftFromStart, shiftFromEnd };
}

struct samtools::CigarShift samtools::hardClipCigar(sam_record & e_sam_record, int & start, int & stop)
{
	unordered_map<int, struct bgitools::CIGAR> new_cigar_map;
	int cigar_ind = 0;
	int index = 0;
	int totalHardClipCount = stop - start + 1;
	int alignmentShift = 0; // GATK:caused by hard clipping deletions

	// GATK:hard clip the beginning of the cigar string
	if (start == 0)
	{
		// GATK:Skip all leading hard clips
		unordered_map<int, struct bgitools::CIGAR> cigar_map = bgitools::get_cigar(e_sam_record);
		int iii;
		for (iii = 0; iii < cigar_map.size(); iii++)
		{
			struct bgitools::CIGAR elt = cigar_map[iii];
			if (elt.opt == bgitools::OPT::H)
			{
				totalHardClipCount = elt.length;
				if (iii + 1 >= cigar_map.size())
				{
					cout << "Read is entirely hardclipped, shouldn't be trying to clip it's cigar string" << endl;
					assert(iii + 1 < cigar_map.size());
				}
			}
			else
			{
				break;
			}
		}

		// GATK:keep clipping until we hit stop
		struct bgitools::CIGAR elt = cigar_map[iii];
		while (index <= stop)
		{
			int shift = 0;
			if (bgitools::cigarOperator(elt.opt).consumesReadBases)
				shift = elt.length;

			// GATK:we're still clipping or just finished perfectly
			if (index + shift == stop + 1)
			{
				alignmentShift += samtools::calculateHardClippingAlignmentShift(elt, elt.length);
				new_cigar_map[cigar_ind] = bgitools::CIGAR{ totalHardClipCount + alignmentShift, bgitools::OPT::H };
				cigar_ind++;
			}
			// GATK:element goes beyond what we need to clip
			else if (index + shift > stop + 1) {
				int elementLengthAfterChopping = elt.length - (stop - index + 1);
				alignmentShift += samtools::calculateHardClippingAlignmentShift(elt, stop - index + 1);
				new_cigar_map[cigar_ind] = bgitools::CIGAR{ totalHardClipCount + alignmentShift, bgitools::OPT::H };
				cigar_ind++;
				new_cigar_map[cigar_ind] = bgitools::CIGAR{ elementLengthAfterChopping, elt.opt };
				cigar_ind++;
			}
			index += shift;
			alignmentShift += samtools::calculateHardClippingAlignmentShift(elt, shift);

			if (index <= stop && (iii + 1<cigar_map.size()))
			{
				iii++;
				elt = cigar_map[iii];
			}
			else
				break;
		}

		// GATK:add the remaining cigar elements
		while (iii + 1<cigar_map.size()) {
			iii++;
			elt = cigar_map[iii];
			new_cigar_map[cigar_ind] = bgitools::CIGAR{ elt.length, elt.opt };
			cigar_ind++;
		}

	}

	// GATK:hard clip the end of the cigar string
	else {
		unordered_map<int, struct bgitools::CIGAR> cigar_map = bgitools::get_cigar(e_sam_record);
		int iii = 0;
		struct bgitools::CIGAR elt = cigar_map[iii];

		// GATK:Keep marching on until we find the start
		while (index < start) {
			int shift = 0;
			if (bgitools::cigarOperator(elt.opt).consumesReadBases)
				shift = elt.length;

			// GATK:we haven't gotten to the start yet, keep everything as is.
			if (index + shift < start)
			{
				new_cigar_map[cigar_ind] = bgitools::CIGAR{ elt.length, elt.opt };
				cigar_ind++;
			}

			// GATK:element goes beyond our clip starting position
			else {
				int elementLengthAfterChopping = start - index;
				alignmentShift += samtools::calculateHardClippingAlignmentShift(elt, elt.length - (start - index));

				// GATK:if this last element is a HARD CLIP operator, just merge it with our hard clip operator to be added later
				if (elt.opt == bgitools::OPT::H)
					totalHardClipCount += elementLengthAfterChopping;
				// otherwise, maintain what's left of this last operator
				else
				{
					new_cigar_map[cigar_ind] = bgitools::CIGAR{ elementLengthAfterChopping, elt.opt };
					cigar_ind++;
				}

			}
			index += shift;
			if (index < start && (iii + 1 < cigar_map.size()))
			{
				iii++;
				elt = cigar_map[iii];
			}
			else
				break;
		}

		while (iii + 1 < cigar_map.size())
		{
			iii++;
			elt = cigar_map[iii];
			alignmentShift += samtools::calculateHardClippingAlignmentShift(elt, elt.length);

			// if the read had a HardClip operator in the end, combine it with the Hard Clip we are adding
			if (elt.opt == bgitools::OPT::H)
				totalHardClipCount += elt.length;
		}
		new_cigar_map[cigar_ind] = bgitools::CIGAR{ totalHardClipCount + alignmentShift, bgitools::OPT::H };
	}
	return samtools::cleanHardClippedCigar(new_cigar_map);
}

void samtools::hardClip(sam_record & e_sam_record, int & start, int & stop)
{
	uint32_t alignmentEnd = bgitools::getAlignmentEnd(e_sam_record);
	unordered_map<int, struct bgitools::CIGAR> cigar_map = bgitools::get_cigar(e_sam_record);
	int softStart = samtools::getSoftStart(cigar_map, e_sam_record.pos);
	int softEnd = samtools::getSoftEnd(cigar_map, alignmentEnd);
	int firstBaseAfterSoftClips = e_sam_record.pos - softStart;
	int lastBaseBeforeSoftClips = softEnd - softStart;

	if (start == firstBaseAfterSoftClips && stop == lastBaseBeforeSoftClips)
	{
		// GATK:note that if the read has no soft clips, these constants will be 0 and read length - 1 (beauty of math).
		e_sam_record.cigar = "";
		e_sam_record.seq = "";
		return;
	}

	// GATK:If the read is unmapped there is no Cigar string and neither should we create a new cigar string
	CigarShift cigarShift = samtools::getReadUnmappedFlag(e_sam_record) ?
		samtools::CigarShift{ unordered_map<int, struct bgitools::CIGAR>{}, 0, 0 } :
		hardClipCigar(e_sam_record, start, stop);

	// GATK:the cigar may force a shift left or right (or both) in case we are left with insertions
	// GATK:starting or ending the read after applying the hard clip on start/stop.
	int newLength = e_sam_record.seq.length() - (stop - start + 1) - cigarShift.shiftFromStart - cigarShift.shiftFromEnd;
	string newBases = "";
	string newQuals = "";
	int copyStart = (start == 0) ? stop + 1 + cigarShift.shiftFromStart : cigarShift.shiftFromStart;

	for (int i = 0; i < e_sam_record.seq.size(); i++)
	{
		if (i >= copyStart && i <= copyStart + newLength - 1)
		{
			newBases += e_sam_record.seq[i];
		}
	}
	for (int i = 0; i < e_sam_record.seq.size(); i++)
	{
		if (i >= copyStart && i <= copyStart + newLength - 1)
		{
			newQuals += e_sam_record.qual[i];
		}
	}

	e_sam_record.qual = newQuals;
	e_sam_record.seq = newBases;
	e_sam_record.cigar_map = cigarShift.cigar_map;
	if (start == 0)
		e_sam_record.pos = e_sam_record.pos + samtools::calculateAlignmentStartShift(cigar_map, cigarShift.cigar_map);
	// TODO: implement it
}

bool samtools::getReadUnmappedFlag(sam_record & e_sam_record)
{
	return ((e_sam_record.flag & globalB::READ_UNMAPPED_FLAG) != 0);
}


void samtools::hardClipByReferenceCoordinates(sam_record & e_sam_record, int & refStart, int & refStop)
{
	if (e_sam_record.seq.size() == 0)
	{
		cout << "sam seq size = 0";
		assert(e_sam_record.seq.size() != 0);
		return;
	}

	int start;
	int stop;

	// GATK:Determine the read coordinate to start and stop hard clipping
	if (refStart < 0)
	{
		if (refStop < 0)
		{
			cout << "Only one of refStart or refStop must be < 0, not both " << refStart << ", " << refStop << ")" << endl;
			assert(refStop >= 0);
		}
		start = 0;
		bool allowGoalNotReached = false;
		stop = grptools::getReadCoordinateForReferenceCoordinate(e_sam_record, refStop, samtools::ClippingTail::LEFT_TAIL, allowGoalNotReached);
	}
	else
	{
		if (refStop >= 0)
		{
			cout << "Either refStart or refStop must be < 0 (" << refStart << ", " << refStop << ")" << endl;
			assert(refStop < 0);
		}
		bool allowGoalNotReached = false;
		start = grptools::getReadCoordinateForReferenceCoordinate(e_sam_record, refStart, samtools::ClippingTail::RIGHT_TAIL, allowGoalNotReached);
		stop = e_sam_record.seq.length() - 1;
	}

	if (start < 0 || stop > e_sam_record.seq.length() - 1)
	{
		cout << "Trying to clip before the start or after the end of a read" << endl;
		assert(start >= 0 && stop > e_sam_record.seq.length() - 1);
	}

	if (start > stop)
	{
		printf("START (%d) > (%d) STOP -- this should never happen -- call Mauricio!", start, stop);
		assert(start <= stop);
	}

	if (start > 0 && stop < e_sam_record.seq.length() - 1)
	{
		printf("Trying to clip the middle of the read: start %d, stop %d, cigar: %s", start, stop, e_sam_record.cigar.c_str());
		assert(!(start > 0 && stop < e_sam_record.seq.length() - 1));
	}

	int readLength = e_sam_record.seq.length();
	if (start < readLength)
	{
		if (stop >= readLength)
		{
			stop = readLength - 1;
		}
		//        string bases = e_sam_record.seq;
		//        string newBases = "";
		//        string newQuals = "";
	}
	hardClip(e_sam_record, start, stop);
}

void samtools::hardClipAdaptorSequence(sam_record & e_sam_record)
{
	int adaptorBoundary = samtools::getAdaptorBoundary(e_sam_record);

	// second condition: !isInsideRead
	if (adaptorBoundary == -1 || !(((adaptorBoundary >= e_sam_record.pos) && (adaptorBoundary <= bgitools::getAlignmentEnd(e_sam_record)))))
	{
		return;
	}

	int StartOrStop = -1;
	((e_sam_record.flag & globalB::READ_STRAND_FLAG) != 0) ?
		samtools::hardClipByReferenceCoordinates(e_sam_record, StartOrStop, adaptorBoundary) :
		samtools::hardClipByReferenceCoordinates(e_sam_record, adaptorBoundary, StartOrStop);
}

void samtools::clipRead(sam_record & e_sam_record, vector<pair<int, int>> & op)
{
	if (op.size() == 0)
	{
		return;
	}

	for (auto e_op : op)
	{
		int readLength = e_sam_record.seq.size();
		// GATK:check if the clipped read can still be clipped in the range requested
		if (e_op.first < readLength)
		{
			pair<int, int> fixedOperation = e_op;
			if (e_op.second >= readLength)
			{
				fixedOperation.second = readLength - 1;
			}
			samtools::hardClip(e_sam_record, fixedOperation.first, fixedOperation.second);
		}
	}

	assert(e_sam_record.seq.size() != 0);
}

void samtools::hardClipSoftClippedBases(sam_record & e_sam_record)
{
	samtools::hardClipAdaptorSequence(e_sam_record);
	if (e_sam_record.seq.length() == 0)
	{
		return;
	}

	int readIndex = 0;
	int cutLeft = -1;            // GATK:first position to hard clip (inclusive)
	int cutRight = -1;           // GATK:first position to hard clip (inclusive)
	bool rightTail = false;   // GATK:trigger to stop clipping the left tail and start cutting the right tail

	unordered_map<int, struct bgitools::CIGAR> cigar_map = bgitools::get_cigar(e_sam_record);
	for (int iii = 0; iii < cigar_map.size(); iii++)
	{
		struct bgitools::CIGAR elt = cigar_map[iii];
		if (elt.opt == bgitools::OPT::S)
		{
			if (rightTail)
			{
				cutRight = readIndex;
			}
			else
			{
				cutLeft = readIndex + elt.length - 1;
			}
		}
		else if (elt.opt != bgitools::OPT::H)
			rightTail = true;

		if (bgitools::cigarOperator(elt.opt).consumesReadBases)
			readIndex += elt.length;
	}

	// GATK:It is extremely important that we cut the end first otherwise the read coordinates change.
	vector<pair<int, int>> op = {};
	if (cutRight >= 0)
	{
		pair<int, int> op1;
		op1.first = cutRight;
		op1.second = e_sam_record.seq.length() - 1;
		op.push_back(op1);
	}
	if (cutLeft >= 0)
	{
		pair<int, int> op2;
		op2.first = 0;
		op2.second = cutLeft;
		op.push_back(op2);
	}

	clipRead(e_sam_record, op);
}

string samtools::sam_record_to_string(map<string, vector<sam_record*>>& map_vec_sam_record)
{
	string sb("");  // string buffer

	for (auto & rname : globalB::vec_chr)
	{
		for (auto & e_sam_record : map_vec_sam_record[rname])
		{
			sb += (*e_sam_record).to_string();
		}
	}
	return sb;
}

// samtools__ end



//#include "grptools.cpp"
// grptools__ start
pair<int, bool>  grptools::getReadCoordinateForReferenceCoordinate(int & alignmentStart, unordered_map<int, struct bgitools::CIGAR> & cigar_map, int & refCoord, const samtools::ClippingTail & tail, bool & allowGoalNotReached)
{
	int readBases = 0;
	int refBases = 0;
	bool fallsInsideDeletion = false;

	int goal = refCoord - alignmentStart;  // GATK:The goal is to move this many reference bases
	if (goal < 0)
	{
		if (allowGoalNotReached)
		{
			return pair<int, bool>(globalB::CLIPPING_GOAL_NOT_REACHED, false);
		}
		else {
			cout << "Somehow the requested coordinate is not covered by the read. Too many deletions?" << endl;
			assert(allowGoalNotReached == true);
		}
	}
	bool goalReached = refBases == goal;


	for (int i = 0; i < cigar_map.size(); i++)
	{
		struct bgitools::CIGAR elt = cigar_map[i];
		if (!goalReached)
		{
			int shift = 0;
			if (bgitools::cigarOperator(elt.opt).consumesReferenceBases ||
					elt.opt == bgitools::OPT::S) // GATK:bgitools::OPT:S = CigarOperator.SOFT_CLIP
			{
				if (refBases + elt.length < goal)
					shift = elt.length;
				else
					shift = goal - refBases;

				refBases += shift;
			}
			goalReached = refBases == goal;

			if (!goalReached && bgitools::cigarOperator(elt.opt).consumesReadBases)
			{
				readBases += elt.length;
			}

			if (goalReached)
			{
				// GATK:Is this base's reference position within this cigar element? Or did we use it all?
				bool endsWithinCigar = shift < elt.length;

				// GATK:If it isn't, we need to check the next one. There should *ALWAYS* be a next one
				// since we checked if the goal coordinate is within the read length, so this is just a sanity check.
				// ls: I think this code has no sense and not necessary.
				if (!endsWithinCigar && (i + 1 >= cigar_map.size())) {
					if (allowGoalNotReached)
					{
						return pair<int, bool>(globalB::CLIPPING_GOAL_NOT_REACHED, false);
					}
					else {
						cout << "Reference coordinate corresponds to a non-existent base in the read."
							" This should never happen -- call Mauricio" << endl;
						assert(allowGoalNotReached == true);
					}
				}

				// if we end inside the current cigar element, we just have to check if it is a deletion
				if (endsWithinCigar)
				{
					fallsInsideDeletion = elt.opt == bgitools::OPT::D;
				} // DELETION;
				else
				{
					// if (i+1 < cigar_map.size())
					struct bgitools::CIGAR nextElt = cigar_map[i + 1];
					i++;
					// GATK:if we end outside the current cigar element, we need to check if the next element is an insertion or deletion.

					// if it's an insertion, we need to clip the whole insertion before looking at the next element
					if (nextElt.opt == bgitools::OPT::I) // INSERTION
					{
						readBases += nextElt.length;
						if (i + 1 >= cigar_map.size())
						{
							if (allowGoalNotReached)
							{
								return pair<int, bool>(globalB::CLIPPING_GOAL_NOT_REACHED, false);
							}
							else
							{
								cout << "Reference coordinate corresponds to a non-existent base in the read. This should never happen -- call Mauricio" << endl;
								assert(allowGoalNotReached == true);
							}
						}

						nextElt = cigar_map[i + 1];
						i++;
					}

					// GATK:if it's a deletion, we will pass the information on to be handled downstream.
					fallsInsideDeletion = nextElt.opt == bgitools::OPT::D;
				}

				// GATK:If we reached our goal outside a deletion, add the shift
				if (!fallsInsideDeletion && bgitools::cigarOperator(elt.opt).consumesReadBases)
					readBases += shift;

				// GATK:If we reached our goal inside a deletion, but the deletion is the next cigar element then we need
				// to add the shift of the current cigar element but go back to it's last element to return the last
				// base before the deletion (see warning in function contracts)
				else if (fallsInsideDeletion && !endsWithinCigar)
					readBases += shift - 1;

				// GATK:If we reached our goal inside a deletion then we must backtrack to the last base before the deletion
				else if (fallsInsideDeletion && endsWithinCigar)
					readBases--;
			}
		}
		else
		{
			break;
		}
	}

	if (!goalReached)
	{
		if (allowGoalNotReached)
		{
			return pair<int, bool>(globalB::CLIPPING_GOAL_NOT_REACHED, false);
		}
		else {
			cout << "Somehow the requested coordinate is not covered by the read. Alignment " << alignmentStart << endl;
			assert(allowGoalNotReached == true);
		}
	}

	return pair<int, bool>(readBases, fallsInsideDeletion);
}

vector<pair<int, int>> grptools::getBindings(sam_record & e_sam_record, vector<globalB::vcf_range*> & e_vec_vcf_overlap_p_range, vector<pair<int, int>> &vec_ft_it, std::vector<std::list<int>> & currentFeatures)
{

	int start = e_sam_record.pos;
	int end = bgitools::getAlignmentEnd(e_sam_record);
	//cout << e_sam_record.cigar << endl;
	//cout << end - start << endl;


	for (int i = 0; i < 1; i++)  //only 1 vcf file now !!!
	{
#if 1
		list<int>::iterator lit = currentFeatures[i].begin();

		while (lit != currentFeatures[i].end())
		{
			//if (map_features[i][*lit].second < start)
			globalB::vcf_range &e_vcf_range = *(e_vec_vcf_overlap_p_range[*lit]);

			if (e_vcf_range.pos + e_vcf_range.len - 1 < start)
			{
				currentFeatures[i].erase(lit++);
			}
			else
			{
				lit++;
			}
		}
#endif

		int *ft_start = &vec_ft_it[i].first;
		while ((*ft_start) != vec_ft_it[i].second)
		{
			//pair<int, int> ftPair = map_features[i][*ft_start];
			auto &ftPair = *(e_vec_vcf_overlap_p_range[*ft_start]);
			//Trick: minus one to deal with possible snp order: {10292359, 10292361}, {10292359, 10292359}
			//TODO: snp order may have other forms

			auto ftPair_end = ftPair.pos + ftPair.len - 1;
			if (ftPair_end < start)
			{
				(*ft_start)++;
			}
			//else if (ftPair.first > end)
			else if (ftPair.pos > end)
			{
				break;
			}
			else if (!(ftPair.pos > end || ftPair_end < start)) //GATK:!(//this.start > that.stop || that.start > this.stop)
			{
				currentFeatures[i].push_back(*ft_start);
				(*ft_start)++;
			}
		}
	}

#if 1

	vector<pair<int, int>> bindings;
	int n_vcf = 0;
	for (auto & e_list : currentFeatures)
	{
		for (auto & e_int : e_list)
		{
			globalB::vcf_range &e_pos_len = *(e_vec_vcf_overlap_p_range[e_int]);
			int e_end = e_pos_len.pos + e_pos_len.len - 1;
			if (!(e_pos_len.pos > end || e_end < start))
			{
				bindings.push_back(make_pair(e_pos_len.pos, e_end));
			}
		}
		n_vcf++;
	}
#endif
	return bindings;
}



vector<bool> grptools::calculateKnownSitesByFeatures(sam_record & e_sam_record, vector<pair<int, int>> & bindings)
{
	int readLength = e_sam_record.seq.length();
	vector<bool> knownSites(readLength, false);
	bool allowGoalNotReached = true;
	for (auto &f : bindings)
	{
		int featureStartOnRead = grptools::getReadCoordinateForReferenceCoordinate(e_sam_record,
				f.first,
				samtools::ClippingTail::LEFT_TAIL,
				allowGoalNotReached); // GATK:BUGBUG: should I use LEFT_TAIL here?
		if (featureStartOnRead == globalB::CLIPPING_GOAL_NOT_REACHED) {
			featureStartOnRead = 0;
		}

		int featureEndOnRead = grptools::getReadCoordinateForReferenceCoordinate(e_sam_record,
				f.second,
				samtools::ClippingTail::LEFT_TAIL,
				allowGoalNotReached);
		if (featureEndOnRead == globalB::CLIPPING_GOAL_NOT_REACHED) {
			featureEndOnRead = readLength;
		}

		if (featureStartOnRead > readLength) {
			featureStartOnRead = featureEndOnRead = readLength;
		}

		int f_start = max(0, featureStartOnRead);
		int f_end = min(readLength, featureEndOnRead + 1);

		for (int i = 0; i < readLength; i++)
		{
			if (i >= f_start && i < f_end)
			{
				knownSites[i] = true;
			}
		}
	}
	return knownSites;
}

vector<sam_record*> grptools::filterPreprocess_v1(int & start, int & end, vector<sam_record>& vec_line_sam_record, vector<bed_record> & vec_bed_record)
{
	vector<sam_record*> filtered_vec_sam_record{};
	for (int j = start; j < end; j++)
	{
		//auto &_e_line = vec_line[j];
		//sam_record e_sam_record(_e_line);
		auto &e_sam_record = vec_line_sam_record[j];
		//vec_line_sam_record[j] = sam_record(_e_line);
		//cout << vec_line_sam_record[j].rname << " ";

		if (bgitools::sam_filter(e_sam_record) || bgitools::bed_filter_v1(e_sam_record, vec_bed_record))
		{
			continue;
		}

		if (e_sam_record.rname == "*")
		{
			cout << "add " << e_sam_record.qname << "|" << e_sam_record.flag << "with chromosome *" << endl;
			assert(e_sam_record.rname != "*");
		}
		filtered_vec_sam_record.push_back(&e_sam_record);
	}
	return filtered_vec_sam_record;
}

vector<sam_record*> grptools::aggregateReadData(vector<sam_record*> & vec_p_sam_record, string &rname, vector<globalB::vcf_range*>& e_vec_vcf_overlap_p_range, string& hg19_path, string& fn_vcf)
{
	auto &first_sam_record = *(vec_p_sam_record[0]);
	auto &tail_sam_record = *(vec_p_sam_record.back());


	int start = first_sam_record.pos;
	int end = bgitools::getAlignmentEnd(tail_sam_record);

	for (int ii = 0; ii < vec_p_sam_record.size() - 1; ii++)
	{
		auto &e_sam_record = *(vec_p_sam_record[ii]);
		int end_tmp = bgitools::getAlignmentEnd(e_sam_record);
		if (end < end_tmp) end = end_tmp;
	}

	vector<sam_record*> final_vec_sam_record;

	const float ratio_max = 0.33;
	int max_num_vcf_ret = ratio_max * (end - start);

	auto sz_range = globalB::getFeaturesByHtslib(globalB::arr_vcf_range_R0, start, end, e_vec_vcf_overlap_p_range);


	//cout << "tabix " + fn_vcf + ".gz "  + rname + ":" << start << "-" << end << endl;

	vector<pair<int, int>> vec_ft_it;
	std::vector<std::list<int>> currentFeatures = {};


	for (int i = 0; i < 1; i++)  //only one vcf is supported now !!!
	{
		pair<int, int> pairIt;
		pairIt.first = 0;
		pairIt.second = (int)sz_range;

		vec_ft_it.push_back(pairIt);

		std::list<int> featuresInd = {};
		currentFeatures.push_back(featuresInd);
	}


#if !CacheChromosome
	faidx_t *hg19fai = fai_load(hg19_path.c_str());
	if (!hg19fai)
	{
		fprintf(stderr, "Could not load fai index of %s\n", hg19_path.c_str());
		assert(0 == 1);
	}
#endif

	for (auto &p_e_filtered_sam_record : vec_p_sam_record)
	{
		auto &e_filtered_sam_record = *p_e_filtered_sam_record;
		e_filtered_sam_record.bindings = grptools::getBindings(e_filtered_sam_record, e_vec_vcf_overlap_p_range, vec_ft_it, currentFeatures);

		samtools::hardClipSoftClippedBases(e_filtered_sam_record);
		if (e_filtered_sam_record.seq.length() == 0)
		{
			continue;
		}
#if CacheChromosome
		//e_sam_record.refSeq = bgitools::safeGetRefSeq_v1(e_sam_record, globalB::chromosome);
		e_filtered_sam_record.refSeq = bgitools::safeGetRefSeq_v1(e_filtered_sam_record, globalB::chromosome);
#else
		e_filtered_sam_record.refSeq = bgitools::safeGetRefSeq(e_filtered_sam_record, hg19fai);
#endif

		final_vec_sam_record.push_back(&e_filtered_sam_record);
	}
#if !CacheChromosome
	fai_destroy(hg19fai);
#endif



	return final_vec_sam_record;
}


string grptools::Arguments::to_string()
{
	const int MAX_CHAR_E_LINE = 115 * 2;
	char e_line[MAX_CHAR_E_LINE] = { 0 };

	const char *title =
		"#:GATKReport.v1.1:5\n"
		"#:GATKTable:2:16:%s:%s:;\n"
		"#:GATKTable:Arguments:Recalibration argument collection values used in this run\n";

	string sb(title);
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "Argument", "Value");
	sb += e_line;

	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "binary_tag_name", binary_tag_name.c_str());                         sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "covariate", covariate.c_str());                                     sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "default_platform", default_platform.c_str());                       sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "force_platform", force_platform.c_str());                           sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "indels_context_size", indels_context_size.c_str());                 sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "insertions_default_quality", insertions_default_quality.c_str());   sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "low_quality_tail", low_quality_tail.c_str());                       sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "mismatches_context_size", mismatches_context_size.c_str());         sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "mismatches_default_quality", mismatches_default_quality.c_str());   sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "no_standard_covs", no_standard_covs.c_str());                       sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "plot_pdf_file", plot_pdf_file.c_str());                             sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "quantizing_levels", quantizing_levels.c_str());                     sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "recalibration_report", recalibration_report.c_str());               sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "run_without_dbsnp", run_without_dbsnp.c_str());                     sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "solid_nocall_strategy", solid_nocall_strategy.c_str());             sb += e_line;
	snprintf(e_line, MAX_CHAR_E_LINE, "%-28s%-72s\n", "solid_recal_mode", solid_recal_mode.c_str());                       sb += e_line;

	return sb;
}

string grptools::QualityScore::to_string(const long &key, const char* record_format, char* e_line, const int & MAX_CHAR_E_LINE)
{
	snprintf(e_line, MAX_CHAR_E_LINE, record_format,
			bgitools::to_string(int(key)).c_str(),
			bgitools::to_string(count).c_str(),
			bgitools::to_string(quantizedScore).c_str());

	return string(e_line);
}

string  grptools::RecalTable0::to_string(string & ReadGroup, string & EventType, const char * record_format, char* e_line, const int & MAX_CHAR_E_LINE)
{
	string sb("");  // string buffer
	empiricalQuality = grptools::getEmpiricalQuality(empiricalQuality, observations, numMismatches);

	snprintf(e_line, MAX_CHAR_E_LINE, record_format,
			ReadGroup.c_str(),
			EventType.c_str(),
			empiricalQuality, estimatedQReported, observations, numMismatches);

	sb += e_line;
	return sb;
}

string grptools::RecalTable1::to_string(string & ReadGroup, int & key, string & EventType, const char * record_format, char* e_line, const int & MAX_CHAR_E_LINE)
{
	string sb("");  // string buffer
	empiricalQuality = grptools::getEmpiricalQuality(empiricalQuality, observations, numMismatches);

	snprintf(e_line, MAX_CHAR_E_LINE, record_format,
			ReadGroup.c_str(),
			key,
			EventType.c_str(),
			empiricalQuality, observations, numMismatches);

	sb += e_line;
	return sb;
}

string grptools::RecalTable2::to_string(string &ReadGroup, RT2KEY &key, string &EventType, const char* record_format, char* e_line, const int & MAX_CHAR_E_LINE)
{
	string sb("");  // string buffer
	string cV = (key.covariateIdx == 0 ?
			grptools::contextFromKey(key.covariateKey) :
			grptools::formatKey(key.covariateKey));
	auto *QualityScore = bgitools::to_string(key.quality).c_str();
	auto *CovariateValue = cV.c_str();
	auto *CovariateName = key.covariateIdx == 0 ? "Context" : "Cycle";

	empiricalQuality = grptools::getEmpiricalQuality(empiricalQuality, observations, numMismatches);

	if (key.quality == 26 && cV == "GC" && key.covariateIdx == 0) // key.covariateIdx == 1 <=> Cycle
	{
		cout << "" << endl;
		if (numMismatches == 0.125)
		{
			cout << numMismatches << endl;
			cout << bgitools::round_(numMismatches + 1e-4, 2) << endl;
			printf("-printf %10.2f\n", numMismatches);
		}
	}



	snprintf(e_line, MAX_CHAR_E_LINE, record_format,
			ReadGroup.c_str(),
			QualityScore, CovariateValue, CovariateName,
			EventType.c_str(),
			empiricalQuality, observations, numMismatches + 1e-4);

	sb += e_line;
	return sb;
}


unordered_map<int, struct grptools::CVKEY> grptools::ComputeCovariates(sam_record & e_sam_record)
{
	int readLength = e_sam_record.seq.length();
	int readOrderFactor = samtools::getReadPairedFlag(e_sam_record.flag) && samtools::getSecondOfPairFlag(e_sam_record.flag) ? -1 : 1;

	string bases = samtools::clipLowQualEnds(e_sam_record, globalB::LOW_QUAL_TAIL, samtools::ClippingRepresentation::WRITE_NS).seq;
	bool negativeStrand = samtools::getReadNegativeStrandFlag(e_sam_record);
	int increment;
	int cycle;
	if (negativeStrand)
	{
		bases = samtools::simpleReverseComplement(bases);
		cycle = readLength * readOrderFactor;
		increment = -1 * readOrderFactor;
	}
	else
	{
		cycle = readOrderFactor;
		increment = readOrderFactor;
	}
	vector<int> mismatchKeys = globalB::contextWith(bases, globalB::mismatchesContextSize, globalB::mismatchesKeyMask);

	unordered_map<int, struct grptools::CVKEY> readCovariates = {};
	struct grptools::CVKEY cvkey; // first: qual, second: context value, last: cycle value.
	for (int i = 0; i < readLength; i++)
	{
		cvkey.quality = e_sam_record.qual[i] - 33;
		cvkey.contextKey = mismatchKeys[(negativeStrand ? readLength - i - 1 : i)];

		if (cycle > globalB::MAXIMUM_CYCLE_VALUE)
		{
			cout << "a larger cycle was detected in read " << e_sam_record.qname << ";" << e_sam_record.flag << endl;
			assert(0 == 1);
		}
		int substitutionKey = samtools::keyFromCycle(cycle);
		// TODO: implement : final int indelKey = (i < CUSHION_FOR_INDELS || i > MAX_CYCLE_FOR_INDELS) ? -1 : substitutionKey;
		cvkey.cycleKey = substitutionKey;
		cycle += increment;

		readCovariates[i] = cvkey;
	}

	return readCovariates;
}

vector<struct grptools::CVKEY> grptools::ComputeCovariates_v1(sam_record & e_sam_record)
{
	int readLength = e_sam_record.seq.length();
	int readOrderFactor = samtools::getReadPairedFlag(e_sam_record.flag) && samtools::getSecondOfPairFlag(e_sam_record.flag) ? -1 : 1;

	string bases = samtools::clipLowQualEnds(e_sam_record, globalB::LOW_QUAL_TAIL, samtools::ClippingRepresentation::WRITE_NS).seq;
	bool negativeStrand = samtools::getReadNegativeStrandFlag(e_sam_record);
	int increment;
	int cycle;
	if (negativeStrand)
	{
		bases = samtools::simpleReverseComplement(bases);
		cycle = readLength * readOrderFactor;
		increment = -1 * readOrderFactor;
	}
	else
	{
		cycle = readOrderFactor;
		increment = readOrderFactor;
	}
	vector<int> mismatchKeys = globalB::contextWith(bases, globalB::mismatchesContextSize, globalB::mismatchesKeyMask);

	vector<struct CVKEY> readCovariates = {};
	struct grptools::CVKEY cvkey; // first: qual, second: context value, last: cycle value.
	for (int i = 0; i < readLength; i++)
	{
		cvkey.quality = e_sam_record.qual[i] - 33;
		cvkey.contextKey = mismatchKeys[(negativeStrand ? readLength - i - 1 : i)];

		if (cycle > globalB::MAXIMUM_CYCLE_VALUE)
		{
			cout << "a larger cycle was detected in read " << e_sam_record.qname << ";" << e_sam_record.flag << endl;
			assert(0 == 1);
		}
		int substitutionKey = samtools::keyFromCycle(cycle);
		// TODO: implement : final int indelKey = (i < CUSHION_FOR_INDELS || i > MAX_CYCLE_FOR_INDELS) ? -1 : substitutionKey;
		cvkey.cycleKey = substitutionKey;
		cycle += increment;

		readCovariates.push_back(cvkey);
	}

	return readCovariates;
}

void grptools::ComputeCovariates_v2(sam_record & e_sam_record, grptools::grpTempDat * grpTempDatArr, int & idx)
{
	int readLength = (int)e_sam_record.seq.length();
	int readOrderFactor = samtools::getReadPairedFlag(e_sam_record.flag) && samtools::getSecondOfPairFlag(e_sam_record.flag) ? -1 : 1;
	//    sam_record clippedRead = samtools::clipLowQualEnds(e_sam_record, B::LOW_QUAL_TAIL, samtools::ClippingRepresentation::WRITE_NS);

	string bases = samtools::clipLowQualEnds(e_sam_record, globalB::LOW_QUAL_TAIL, samtools::ClippingRepresentation::WRITE_NS).seq;
	bool negativeStrand = samtools::getReadNegativeStrandFlag(e_sam_record);
	int increment;
	int cycle;
	if (negativeStrand)
	{
		bases = samtools::simpleReverseComplement(bases);
		cycle = readLength * readOrderFactor;
		increment = -1 * readOrderFactor;
	}
	else
	{
		cycle = readOrderFactor;
		increment = readOrderFactor;
	}
	//	cout << e_sam_record.qname << ";" << e_sam_record.flag << endl;
	vector<int> mismatchKeys = globalB::contextWith(bases, globalB::mismatchesContextSize, globalB::mismatchesKeyMask);
	//    int MAX_CYCLE_FOR_INDELS = readLength - B::CUSHION_FOR_INDELS - 1;

	struct grptools::CVKEY * readCovariates = (grptools::CVKEY *)malloc(sizeof(grptools::CVKEY) * readLength);
	assert(NULL != readCovariates);
	struct grptools::CVKEY cvkey; // first: qual, second: context value, last: cycle value.
	for (int i = 0; i < readLength; i++)
	{
		// qual
		readCovariates[i].quality = e_sam_record.qual[i] - 33;

		// context value
		// GATK:final GATKSAMRecord
		readCovariates[i].contextKey = mismatchKeys[(negativeStrand ? readLength - i - 1 : i)];

		// cycle value
		// TODO: implement
		// GATK:final NGSPlatform ngsPlatform = default_platform == null ? read.getNGSPlatform() : NGSPlatform.fromReadGroupPL(default_platform);
		if (cycle > globalB::MAXIMUM_CYCLE_VALUE)
		{
			cout << "a larger cycle was detected in read " << e_sam_record.qname << ";" << e_sam_record.flag << endl;
			assert(0 == 1);
		}
		int substitutionKey = samtools::keyFromCycle(cycle);
		// TODO: implement : final int indelKey = (i < CUSHION_FOR_INDELS || i > MAX_CYCLE_FOR_INDELS) ? -1 : substitutionKey;
		readCovariates[i].cycleKey = substitutionKey;
		cycle += increment;

	}

	grpTempDatArr[idx].readCovariates = readCovariates;
}

vector<bool> grptools::calculateSkipArray(sam_record& e_sam_record, vector<bool> & knownSites)
{
	vector<bool> skip(e_sam_record.seq.size(), 0);
	for (size_t iii = 0; iii < e_sam_record.seq.size(); iii++)
	{
		skip[iii] = ((!bgitools::isRegularBase(e_sam_record.seq[iii])) ||
				bgitools::isLowQualityBase(e_sam_record, iii) ||
				knownSites[iii]);

		//TODO:implement badSolidOffset
	}
	return skip;
}

void grptools::calculateSkipArray_v1(sam_record& e_sam_record, vector<bool> & knownSites, grptools::grpTempDat * grpTempDatArr, int & idx)
{
	bool * skip = (bool *)malloc(sizeof(bool) * e_sam_record.seq.size());
	assert(NULL != skip);
	for (size_t iii = 0; iii < e_sam_record.seq.size(); iii++)
	{
		skip[iii] = 0;
		skip[iii] = ((!bgitools::isRegularBase(e_sam_record.seq[iii])) ||
				bgitools::isLowQualityBase(e_sam_record, iii) ||
				knownSites[iii]);

		//TODO:implement badSolidOffset
		//                grptools::badSolidOffset(e_sam_record, iii);


	}
	grpTempDatArr[idx].skip = skip;
}

vector<int> grptools::calculateIsSNP(sam_record& e_sam_record, string & refSeq)
{
	string readBases = e_sam_record.seq;
	// GATK: final byte[] refBases = Arrays.copyOfRange(ref.getBases(),
	uint32_t readStart = e_sam_record.pos;
	uint32_t readEnd = bgitools::getAlignmentEnd(e_sam_record);
	uint32_t offset = globalB::cb / 2; //ensure offset = 3!!!
	assert(offset == 3);
	uint32_t start = max(int(readStart - offset - samtools::getInsertionOffset(e_sam_record, 0)), 0);
	string refBases = refSeq.substr(readStart - start, readEnd - readStart + 1);

	vector<int> snp(readBases.length(), 0);
	int readPos = 0;
	int refPos = 0;
	unordered_map<int, struct bgitools::CIGAR> cigar_map = bgitools::get_cigar(e_sam_record);
	for (int iii = 0; iii < cigar_map.size(); iii++)
	{
		struct bgitools::CIGAR elt = cigar_map[iii];
		int elementLength = elt.length;
		switch (elt.opt)
		{
			case bgitools::OPT::M:
			case bgitools::OPT::EQ:
			case bgitools::OPT::X:
				for (int iii = 0; iii < elementLength; iii++)
				{
					snp[readPos] = (bgitools::basesAreEqual(readBases[readPos], refBases[refPos]) ? 0 : 1);
					readPos++;
					refPos++;
				}
				break;
			case bgitools::OPT::D:
			case bgitools::OPT::N:
				refPos += elementLength;
				break;
			case bgitools::OPT::I:
			case bgitools::OPT::S: // ReferenceContext doesn't have the soft clipped bases!
				readPos += elementLength;
				break;
			case bgitools::OPT::H:
			case bgitools::OPT::P:
				break;
			default:
				printf("Unsupported cigar operator");
				assert(0 == 1);
		}
	}

	return snp;
}

void grptools::calculateAndStoreErrorsInBlock(int iii, int & blockStartIndex, vector<int> & errorArray, vector<double> & fractionalErrors)
{
	int totalErrors = 0;
	for (int jjj = max(0, blockStartIndex - 1); jjj <= iii; jjj++)
	{
		totalErrors += errorArray[jjj];
	}
	for (int jjj = max(0, blockStartIndex - 1); jjj <= iii; jjj++)
	{
		fractionalErrors[jjj] = ((double)totalErrors) / ((double)(iii - max(0, blockStartIndex - 1) + 1));
	}
}

void grptools::calculateAndStoreErrorsInBlock_v1(int iii, int & blockStartIndex, vector<int> & errorArray, double * fractionalErrors)
{
	int totalErrors = 0;
	for (int jjj = max(0, blockStartIndex - 1); jjj <= iii; jjj++)
	{
		totalErrors += errorArray[jjj];
	}
	for (int jjj = max(0, blockStartIndex - 1); jjj <= iii; jjj++)
	{
		fractionalErrors[jjj] = ((double)totalErrors) / ((double)(iii - max(0, blockStartIndex - 1) + 1));
	}
}

vector<double> grptools::calculateFractionalErrorArray(vector<int> & errorArray, string & baqArray)
{
	// errorArray : snp arr
	// baqArray : baq
	if (errorArray.size() != baqArray.length())
	{
		printf("Array length mismatch detected. Malformed read?");
		assert(0 == 1);
	}

	const char NO_BAQ_UNCERTAINTY = '@';
	const int BLOCK_START_UNSET = -1;

	vector<double> fractionalErrors(baqArray.length(), 0.0); //fractionalErrors.resize(baqArray.length());
	bool inBlock = false;
	int blockStartIndex = BLOCK_START_UNSET;
	int iii;
	for (iii = 0; iii < fractionalErrors.size(); iii++) {
		if (baqArray[iii] == NO_BAQ_UNCERTAINTY) {
			if (!inBlock) {
				fractionalErrors[iii] = (double)errorArray[iii];
			}
			else {
				calculateAndStoreErrorsInBlock(iii, blockStartIndex, errorArray, fractionalErrors);
				inBlock = false; // reset state variables
				blockStartIndex = BLOCK_START_UNSET; // reset state variables
			}
		}
		else {
			inBlock = true;
			if (blockStartIndex == BLOCK_START_UNSET) { blockStartIndex = iii; }
		}
	}
	if (inBlock)
	{
		calculateAndStoreErrorsInBlock(iii - 1, blockStartIndex, errorArray, fractionalErrors);
	}
	if (fractionalErrors.size() != errorArray.size())
	{
		printf("Output array length mismatch detected. Malformed read?");
		assert(0 == 1);
	}
	return fractionalErrors;
}

void grptools::calculateFractionalErrorArray_v1(vector<int> & isSNP, string & baqArray, grptools::grpTempDat * grpTempDatArr, int & idx)
{
	// errorArray : snp arr
	// baqArray : baq
	if (isSNP.size() != baqArray.length())
	{
		printf("Array length mismatch detected. Malformed read?");
		assert(0 == 1);
	}

	const char NO_BAQ_UNCERTAINTY = '@';
	const int BLOCK_START_UNSET = -1;

	//    vector<double> fractionalErrors(baqArray.length(), 0.0); //fractionalErrors.resize(baqArray.length());
	double * fractionalErrors = (double *)malloc(sizeof(double) * baqArray.length());
	assert(NULL != fractionalErrors);
	bool inBlock = false;
	int blockStartIndex = BLOCK_START_UNSET;
	int iii;
	for (iii = 0; iii < baqArray.length(); iii++) {
		fractionalErrors[iii] = 0.0;
		if (baqArray[iii] == NO_BAQ_UNCERTAINTY) {
			if (!inBlock) {
				fractionalErrors[iii] = (double)isSNP[iii];
			}
			else {
				grptools::calculateAndStoreErrorsInBlock_v1(iii, blockStartIndex, isSNP, fractionalErrors);
				inBlock = false; // reset state variables
				blockStartIndex = BLOCK_START_UNSET; // reset state variables
			}
		}
		else {
			inBlock = true;
			if (blockStartIndex == BLOCK_START_UNSET) { blockStartIndex = iii; }
		}
	}
	if (inBlock)
	{
		grptools::calculateAndStoreErrorsInBlock_v1(iii - 1, blockStartIndex, isSNP, fractionalErrors);
	}
	grpTempDatArr[idx].snpErrors = fractionalErrors;
}

double grptools::calcExpectedErrors(const double & estimatedQReported, const double & observations)
{
	return observations * globalB::convertFromPhredScale(estimatedQReported);
}

string grptools::contextFromKey(int & key)
{
	if (key < 0)
	{
		printf("dna conversion cannot handle negative numbers. Possible overflow?");
		assert(key >= 0);
	}
	int length = key & globalB::LENGTH_MASK; // GATK:the first bits represent the length (in bp) of the context
	int mask = 48; // GATK:use the mask to pull out bases
	int offset = globalB::LENGTH_BITS;

	string dna = "";
	for (int i = 0; i < length; i++) {
		int baseIndex = (key & mask) >> offset;
		dna += samtools::baseIndexToSimpleBase(baseIndex);
		mask = mask << 2; // move the mask over to the next 2 bits
		offset += 2;
	}

	return dna;
}

string grptools::formatKey(int & key)
{
	int cycle = key >> 1; // shift so we can remove the "sign" bit
	if ((key & 1) != 0) // is the last bit set?
		cycle *= -1; // then the cycle is negative

	char str[50]; // 50 is ok???
	sprintf(str, "%d", cycle);
	return str;
}

double grptools::getEmpiricalQuality(double & empiricalQuality, double & observations, double & numMismatches)
{
	bool noEQ = fabs(empiricalQuality + 1) < 10e-4;
	if (noEQ)
	{
		double empiricalQual;
		if (observations == 0.)
		{
			empiricalQual = 0.0;
		}
		else
		{
			// GATK:cache the value so we don't call log over and over again
			double doubleMismatches = numMismatches + globalB::SMOOTHING_CONSTANT;
			// GATK:smoothing is one error and one non-error observation, for example
			double doubleObservations = observations + globalB::SMOOTHING_CONSTANT + globalB::SMOOTHING_CONSTANT;
			empiricalQual = doubleMismatches / doubleObservations;
		}
		empiricalQual = -10 * log10(empiricalQual);
		empiricalQuality = min(empiricalQual, (double)globalB::MAX_PHRED_SCORE);
	}
	return empiricalQuality;
}

int grptools::probToQual(long double & prob, long double & eps)
{
	long double lp;
	lp = -10.0*log10(1.0 - prob + eps + pow(10.0, -10));

	long long lp1 = (long long)round(lp);
	assert(fabs(lp - 0.0) > 0);
	long long minVal = 1;
	return (int)max(min(lp1, (long long)globalB::MAX_PHRED_SCORE), minVal);
}

double grptools::getErrorRate(long & nObservations, long & nErrors, int & fixedQual)
{
	if (fixedQual != -1)
	{
		return globalB::qualToErrorProbCache[fixedQual & 0xff];
	}
	else if (nObservations == 0)
	{
		return 0.0;
	}
	else
	{
		return (nErrors + 1) / (1.0 * (nObservations + 1));
	}
}

struct grptools::QualInterval grptools::merge(struct grptools::QualInterval & fromMerge, struct grptools::QualInterval & toMerge)
{
	int compare = (fromMerge.qStart < toMerge.qStart) ? -1 : ((fromMerge.qStart == toMerge.qStart) ? 0 : 1);
	struct grptools::QualInterval left = compare < 0 ? fromMerge : toMerge;
	struct grptools::QualInterval right = compare < 0 ? toMerge : fromMerge;

	if (left.qEnd + 1 != right.qStart)
	{
		printf("Attempting to merge non-continguous intervals");
		assert(left.qEnd + 1 == right.qStart);
	}
	long nCombinedObs = left.nObservations + right.nObservations;
	long nCombinedErr = left.nErrors + right.nErrors;
	int level = max(left.level, right.level) + 1;

	struct grptools::QualInterval merged;
	vector<struct grptools::QualInterval> subIntervals{ left, right };
	merged.qStart = left.qStart;
	merged.qEnd = right.qEnd;
	merged.nObservations = nCombinedObs;
	merged.level = level;
	merged.mergeOrder = 0;
	merged.fixedQual = -1;
	merged.nErrors = nCombinedErr;
	merged.subIntervals = subIntervals;

	return merged;
}

double grptools::getPenalty(struct grptools::QualInterval & e_interval, double & globalErrorRate)
{
	if (globalErrorRate == 0.0) // GATK:there were no observations, so there's no penalty
		return 0.0;

	if (e_interval.subIntervals.empty())
	{
		// GATK:this is leave node
		if (e_interval.qEnd <= globalB::MIN_USABLE_Q_SCORE)
			// GATK:It's free to merge up quality scores below the smallest interesting one
			return 0;
		else {
			return (abs(log10(grptools::getErrorRate(e_interval.nObservations,
								e_interval.nErrors,
								e_interval.fixedQual)) - log10(globalErrorRate))) * e_interval.nObservations;
		}
	}
	else {
		double sum = 0.;
		for (auto &e_subInterval : e_interval.subIntervals)
			sum += grptools::getPenalty(e_subInterval, globalErrorRate);
		return sum;
	}
}

vector<struct grptools::QualInterval> grptools::removeAndAdd(vector<struct grptools::QualInterval> & interval, struct grptools::QualInterval & qualInterval)
{
	assert(qualInterval.subIntervals.size() != 0);

	vector<struct grptools::QualInterval> intervalCopy;
	vector<struct grptools::QualInterval> subIntervals = qualInterval.subIntervals;
	int cnt = 0;
	for (int i = 0; i < interval.size(); i++)
	{
		int itStart, subQstart;
		bool pass = false;
		for (int j = 0; j < subIntervals.size(); j++)
		{
			itStart = interval[i].qStart;
			subQstart = subIntervals[j].qStart;
			if (itStart == subQstart)
			{
				pass = true;
				break;
			}
		}
		if (pass)
		{
			if (cnt < 1)
			{
				cnt += 1;
				intervalCopy.push_back(qualInterval);
			}
		}
		else
		{
			intervalCopy.push_back(interval[i]);
		}
	}

	return intervalCopy;
}

grptools::Arguments::Arguments()
{
	binary_tag_name = string("null");
	covariate = string("ReadGroupCovariate,QualityScoreCovariate,ContextCovariate,CycleCovariate");
	default_platform = string("null");
	force_platform = string("null");
	indels_context_size = string("3");
	insertions_default_quality = string("45");
	low_quality_tail = string("2");
	mismatches_context_size = string("2");
	mismatches_default_quality = string("-1");
	no_standard_covs = string("false");
	plot_pdf_file = string("null");
	quantizing_levels = string("16");
	recalibration_report = string("null");
	run_without_dbsnp = string("false");
	solid_nocall_strategy = string("THROW_EXCEPTION");
	solid_recal_mode = string("SET_Q_ZERO");
}

int grptools::getReadCoordinateForReferenceCoordinate(sam_record & e_sam_record, int & refCoord, const samtools::ClippingTail & tail, bool & allowGoalNotReached)
{
	unordered_map<int, struct bgitools::CIGAR> cigar_map = bgitools::get_cigar(e_sam_record);
	int alignmentStart = samtools::getSoftStart(cigar_map, e_sam_record.pos);

	pair<int, bool> result = grptools::getReadCoordinateForReferenceCoordinate(alignmentStart, cigar_map,
			refCoord, tail, allowGoalNotReached);

	int readCoord = result.first;

	// GATK:Corner case one: clipping the right tail and falls on deletion, move to the next
	// read coordinate. It is not a problem for the left tail because the default answer
	// from getReadCoordinateForReferenceCoordinate is to give the previous read coordinate.
	if (result.second && tail == samtools::ClippingTail::RIGHT_TAIL)
	{
		readCoord++;
	}

	// GATK:clipping the left tail and first base is insertion, go to the next read coordinate
	// with the same reference coordinate. Advance to the next cigar element, or to the
	// end of the read if there is no next element.
	pair<bool, struct bgitools::CIGAR> firstElementIsInsertion = samtools::readStartsWithInsertion(cigar_map);

	if (readCoord == 0 && tail == samtools::ClippingTail::LEFT_TAIL && firstElementIsInsertion.first)
	{
		readCoord = min(firstElementIsInsertion.second.length, samtools::getReadLengthForCigar(cigar_map) - 1);
	}

	return readCoord;
}



// grptools__ end



//#include "globalb.cpp"
// globalB__ start
int globalB::getFeaturesByHtslib(vcf_range* arr_vcf_range, int &start, int &end, vector<globalB::vcf_range*>& vec_overlap_p_range)
{
	const float ratio_max = 0.33; // an experienment number
	const int s_scope = 711;  // the left or right search scope, an experience number

	uint64_t last_s_start = 0;
	uint64_t s_start_0 = start;
	uint64_t s_start_1 = end;

	auto idx_0 = globalB::bin_search_nearest_smaller_idx(arr_vcf_range, globalB::sz_arr_vcf_range_R0, s_start_0, last_s_start);

	auto idx_1 = globalB::bin_search_nearest_smaller_idx(arr_vcf_range, globalB::sz_arr_vcf_range_R0, s_start_1, last_s_start);

	auto delta_idx = idx_1 - idx_0;
	if (delta_idx + 2 * s_scope > vec_overlap_p_range.size())
	{
		cout << " - if ( idx_1 - idx_0 > MAX_OVERLAP), delta =  " << delta_idx << " " << vec_overlap_p_range.size() << endl;
		vec_overlap_p_range.resize(delta_idx + 2 * s_scope);
	}

	int i_start = idx_0 - s_scope;
	int i_end = idx_1 + s_scope;

	if (i_start < 0)
	{
		i_start = 0;
	}

	if (i_end > globalB::sz_arr_vcf_range_R0)
	{
		i_end = globalB::sz_arr_vcf_range_R0;
	}


	int flag_first_meet_range_idx = 0;
	int idx_start_for_next_run = 0;
	int cnt_cmp = 0;

	auto &s_end = s_start_1;
	auto &s_start = s_start_0;
	int cnt_range = 0;



	for (int i = i_start; i < i_end; i++)
	{
		// if they overlap
		auto & e_pos = globalB::arr_vcf_range_R0[i].pos;
		auto & e_len = globalB::arr_vcf_range_R0[i].len;
		auto e_end = e_pos + e_len - 1;


		if (e_pos > s_end)
		{
			break;
		}
		cnt_cmp++;


		// not ($--$ |--|)   or   (|--| $--$) 

		if (!((s_end < e_pos) || (e_end < s_start)))
		{
			vec_overlap_p_range[cnt_range] = globalB::arr_vcf_range_R0 + i;
			if (flag_first_meet_range_idx++ == 0)
			{
				idx_start_for_next_run = i;
				//last_s_start++;
			}
			cnt_range++;

		}
	}



	auto sz_range = cnt_range;
	//cout << "- sz_range: " << sz_range << endl; 
	return sz_range;
}
void globalB::relatedDeltaQInit(grp_tables & gt, int & readLength)
{
	globalB::globalDeltaQ = bqsrtools::calculateGlobalDeltaQ(gt.recalTable0);
	globalB::deltaQCovariatesArr = new double**[globalB::MAX_PHRED_SCORE - globalB::MIN_USABLE_Q_SCORE + 1];
	for (int i = 0; i < globalB::MAX_PHRED_SCORE - globalB::MIN_USABLE_Q_SCORE + 1; i++)
	{
		globalB::deltaQCovariatesArr[i] = new double*[4 * 4 + 1];
		for (int j = 0; j < 4 * 4 + 1; j++)
		{
			globalB::deltaQCovariatesArr[i][j] = new double[readLength + readLength];
		}
	}
	unordered_map<int, int> cycle_k_v_Map;
	for (int ii = 0; ii < readLength + readLength; ii++)
	{
		int cycle, cycleKey;
		if (ii < readLength)
		{
			cycle = ii - readLength;
		}
		if (ii >= readLength)
		{
			cycle = ii - readLength + 1;
		}
		cycleKey = samtools::keyFromCycle(cycle);
		cycle_k_v_Map[ii] = cycleKey;
		globalB::cycle_v_k_Map[cycleKey] = ii;
	}
	unordered_map<int, int> context_k_v_Map;
	vector<string> contextValue{ "AT","AA","AC","AG","CA","CC","CG","CT",
		"GA","GC","GG","GT","TA","TC","TG","TT" };
	for (int i = 0; i < 16; i++)
	{
		int contextKey = samtools::keyFromContext(contextValue[i], 0, globalB::mismatchesContextSize);
		context_k_v_Map[i] = contextKey;
		globalB::context_v_k_Map[contextKey] = i;
	}
	context_k_v_Map[16] = -1;
	globalB::context_v_k_Map[-1] = 16;

	for (int q = 0; q <= globalB::MAX_PHRED_SCORE - globalB::MIN_USABLE_Q_SCORE; q++)
	{
		int qualFromRead = q + globalB::MIN_USABLE_Q_SCORE;
		double deltaQReported = bqsrtools::CalculateDeltaQReported(gt.RecalTable1, globalDeltaQ, qualFromRead);
		map_deltaQReported[qualFromRead] = deltaQReported;
		// 0-99; -50--1, 1-50
		for (int i = 0; i < 4 * 4 + 1; i++)
		{
			int contextValue = context_k_v_Map[i];
			for (int covariate = 0; covariate < readLength + readLength; covariate++)
			{
				int cycleValue = cycle_k_v_Map[covariate];
				vector<int> keySet{ qualFromRead, contextValue, cycleValue };
				double deltaQCovariates = bqsrtools::CalculateDeltaQCovariates(gt.RecalTable2, keySet, globalDeltaQ, deltaQReported);

				globalB::deltaQCovariatesArr[q][i][covariate] = deltaQCovariates;
			}
		}
	}
}

void globalB::relatedDeltaQRelease()
{
	for (int i = 0; i < globalB::MAX_PHRED_SCORE - globalB::MIN_USABLE_Q_SCORE + 1; i++)
	{
		for (int j = 0; j < 4 * 4 + 1; j++)
		{
			DEL_ARR_MEM(deltaQCovariatesArr[i][j]);
		}
		DEL_ARR_MEM(deltaQCovariatesArr[i]);
	}
	DEL_ARR_MEM(deltaQCovariatesArr);
}

double globalB::convertFromPhredScale(const double & x) { return (pow(10.0, (-x) / 10.0)); }

void globalB::allocate_3d_epsilons_1d_qual2prob()
{
	static int run_once_allocate_3d_epsilons_1d_qual2prob_flag = 0;
	if (run_once_allocate_3d_epsilons_1d_qual2prob_flag++ != 0)
	{
		return;
	}

	assert(NULL == globalB::EPSILONS);
	assert(NULL == globalB::qual2prob);

	globalB::EPSILONS = new double**[globalB::eps_3d_shape[0]];
	assert(NULL != globalB::EPSILONS);
	for (int i = 0; i < globalB::eps_3d_shape[0]; i++)
	{
		globalB::EPSILONS[i] = NULL;
		globalB::EPSILONS[i] = new double*[globalB::eps_3d_shape[1]];
		assert(NULL != globalB::EPSILONS[i]);
		for (int j = 0; j < globalB::eps_3d_shape[1]; j++)
		{
			globalB::EPSILONS[i][j] = NULL;
			globalB::EPSILONS[i][j] = new double[globalB::eps_3d_shape[2]];
			assert(NULL != globalB::EPSILONS[i][j]);
		}
	}

	globalB::qual2prob = new double[globalB::eps_3d_shape[0]];
	assert(NULL != globalB::qual2prob);
}

void globalB::release_3d_epsilons_1d_qual2prob()
{

	for (int i = 0; i < globalB::eps_3d_shape[0]; i++)
	{
		for (int j = 0; j < globalB::eps_3d_shape[1]; j++)
		{
			DEL_ARR_MEM(globalB::EPSILONS[i][j]);
		}

		DEL_ARR_MEM(globalB::EPSILONS[i]);
	}

	DEL_ARR_MEM(globalB::EPSILONS);

	DEL_ARR_MEM(globalB::qual2prob);
}

void globalB::initializeCachedData(double *** array3d, double * array1d)
{
	for (int i = 0; i < globalB::eps_3d_shape[0]; i++)
	{
		array1d[i] = pow(10, -i / 10.);

		for (int j = 0; j < globalB::eps_3d_shape[1]; j++)
		{
			for (int q = 0; q < globalB::eps_3d_shape[2]; q++)
			{
				array3d[i][j][q] = 1.0;
			}
		}
	}

	for (char b1 : "ACGTacgt")
	{
		for (char b2 : "ACGTacgt")
		{
			for (int q = 0; q <= globalB::MAX_PHRED_SCORE; q++)
			{
				double qual = qual2prob[q < globalB::minBaseQual ? globalB::minBaseQual : q];
				double e = tolower(b1) == tolower(b2) ? 1 - qual : qual * globalB::EM;

				auto i0 = (int)b1;
				auto i1 = (int)b2;
				auto &i2 = q;

				assert(i0 < globalB::eps_3d_shape[0]);
				assert(i1 < globalB::eps_3d_shape[1]);
				assert(i2 < globalB::eps_3d_shape[2]);

				array3d[i0][i1][i2] = e;
			}
		}
	}
}

int globalB::createMask(const int & contextSize)
{
	int mask = 0;
	// create 2*contextSize worth of bits
	for (int i = 0; i < contextSize; i++)
		mask = (mask << 2) | 3;
	// shift 4 bits to mask out the bits used to encode the length
	return mask << globalB::LENGTH_BITS;
}

vector<int> globalB::contextWith(string & bases, const int & contextSize, const int & mask)
{
	int readLength = bases.length();
	int currentIndex = 0;
	vector<int> keys(readLength, 0);

	// the first contextSize-1 bases will not have enough previous context
	for (int i = 1; i < contextSize && i <= readLength; i++)
	{
		keys[currentIndex] = -1;
		currentIndex++;
	}

	if (readLength < contextSize)
		return keys;

	int newBaseOffset = 2 * (contextSize - 1) + globalB::LENGTH_BITS;

	// get (and add) the key for the context starting at the first base
	int currentKey = samtools::keyFromContext(bases, 0, contextSize);
	keys[currentIndex] = currentKey;
	currentIndex++;


	// if the first key was -1 then there was an N in the context; figure out how many more consecutive contexts it affects
	int currentNPenalty = 0;
	if (currentKey == -1)
	{
		currentKey = 0;
		currentNPenalty = contextSize - 1;
		int offset = newBaseOffset;
		while (bases[currentNPenalty] != 'N')
		{
			int baseIndex = samtools::simpleBaseToBaseIndex(bases[currentNPenalty]);
			currentKey |= (baseIndex << offset);
			offset -= 2;
			currentNPenalty--; // java code, array idx can be negtive number?
		}
	}

	for (; currentIndex < readLength; currentIndex++)
	{
		int baseIndex = samtools::simpleBaseToBaseIndex(bases[currentIndex]);
		if (baseIndex == -1)
		{ // ignore non-ACGT bases
			currentNPenalty = contextSize;
			currentKey = 0; // reset the key
		}
		else
		{
			//TODO: high 4bits record the context bases is redundant
			// push this base's contribution onto the key: shift everything 2 bits, mask out the non-context bits, and add the new base and the length in
			currentKey = (currentKey >> 2) & mask;
			currentKey |= (baseIndex << newBaseOffset);
			currentKey |= contextSize;
		}

		if (currentNPenalty == 0)
		{
			keys[currentIndex] = currentKey;
		}
		else
		{
			currentNPenalty--;
			keys[currentIndex] = -1;
		}
	}

	return keys;
}

vector<double> globalB::qualToErrorProb()
{
	vector<double> prob;
	for (int i = 0; i < 256; i++)
	{
		prob.push_back(globalB::convertFromPhredScale((double)i));
	}

	return prob;
}

void globalB::grp_allocate(int & td_num)
{
	grp_table_arr = new grp_tables[td_num];
	for (int iii = 0; iii < td_num; iii++)
	{
		grp_table_arr[iii].allocateRT2();
	}
}

void globalB::grp_release(int & td_num)
{
	for (int iii = 0; iii < td_num; iii++)
	{
		grp_table_arr[iii].releaseRT2();
	}
	DEL_ARR_MEM(grp_table_arr);
}

void globalB::vcf_record_uniq()
{

	auto &sz = globalB::sz_arr_vcf_range_R0;
	if (sz == 0)
	{
		return;
	}

	int cnt_cp = 0;

	vcf_range* arr_vcf_range_cache = NULL;
	arr_vcf_range_cache = new vcf_range[sz];
	assert(NULL != arr_vcf_range_cache);
	memset(arr_vcf_range_cache, 0, sz * sizeof(vcf_range));

	memcpy(&arr_vcf_range_cache[0], &globalB::arr_vcf_range_R0[0], sizeof(vcf_range) * 1);

	for (int i = 0; i < sz; i++)
	{
		if (
				arr_vcf_range_cache[cnt_cp].pos == globalB::arr_vcf_range_R0[i].pos &&
				arr_vcf_range_cache[cnt_cp].len == globalB::arr_vcf_range_R0[i].len
		   )
		{
		}
		else
		{
			cnt_cp++;
			arr_vcf_range_cache[cnt_cp] = globalB::arr_vcf_range_R0[i];

		}

	}

	//cout << "- cnt_cp:" << cnt_cp << ", sz:" << sz << endl;
	memcpy(globalB::arr_vcf_range_R0, arr_vcf_range_cache, sizeof(vcf_range) * sz);

	delete[] arr_vcf_range_cache;
	arr_vcf_range_cache = NULL;
	//sz = cnt_cp + 1;
	globalB::sz_arr_vcf_range_R0 = cnt_cp + 1;
}

int  globalB::bin_search_nearest_smaller_idx(vcf_range* arr_vcf_range, int& sz, uint64_t& s_start, uint64_t& last_s_start)
{
	int s_cnt = 0;


	auto high = sz - 1;

	if (s_start <= arr_vcf_range[0].pos)
		return 0;

	if (s_start >= arr_vcf_range[high].pos)
		return sz - 1;

	auto low = 0;

	while (1)
	{
		s_cnt++;
		auto index = (high + low) / 2;

		if (arr_vcf_range[index].pos <= s_start)
		{
			low = index;
		}
		else
		{
			high = index;
		}

		if (low >= high - 1)
		{
			//cout << "- bin_search s_cnt: " << s_cnt << endl;
			return low;
		}
	}

}

int globalB::load_vcf_range_by_chrname(string &search_chrname)
{
	//map_chr_range
	if (map_chr_range.find(search_chrname) == map_chr_range.end())
	{
		return 0;
	}

	auto &id_range = map_chr_range[search_chrname];

	assert(NULL != arr_vcf_range_R0);

	auto &if_bin = if_vcf_bin;
	fseek(if_bin, (long)id_range.offset, SEEK_SET);
	fread(arr_vcf_range_R0, 1, (long)id_range.len, if_bin);

	assert(0 == id_range.len % sizeof(vcf_range));
	sz_arr_vcf_range_R0 = (int)id_range.len / sizeof(vcf_range);
	return sz_arr_vcf_range_R0;
}

void globalB::delete_bytes_of_arr_vcf_range()
{
	// close if_vcf_bin
	assert(NULL != if_vcf_bin);
	fclose(if_vcf_bin);
	if_vcf_bin = NULL;

	assert(NULL != arr_vcf_range_R0);
	delete[] arr_vcf_range_R0;
	arr_vcf_range_R0 = NULL;
	assert(NULL == arr_vcf_range_R0);
}

void globalB::alloc_bytes_of_arr_vcf_range(uint64_t &max_len)
{
	assert(NULL == arr_vcf_range_R0);
	arr_vcf_range_R0 = new vcf_range[(long)max_len / sizeof(vcf_range) + 1];
	assert(NULL != arr_vcf_range_R0);
}

map<string, globalB::chr_start_len>& globalB::init_vcf_idx_2_map(string& fn_vcf, uint64_t &max_len)
{
	string fn_vcf_bin = fn_vcf + ".bin";
	string fn_idx = fn_vcf_bin + ".idx";

	assert(NULL == if_vcf_bin);
	if_vcf_bin = fopen(fn_vcf_bin.c_str(), "rb");  assert(NULL != if_vcf_bin);

	// start get map
	ifstream if_vcf_bin_idx(fn_idx.c_str()); assert(if_vcf_bin_idx.is_open());

	globalB::map_chr_range.clear();

	string e_line;

	while (!if_vcf_bin_idx.eof())
	{

		std::getline(if_vcf_bin_idx, e_line);

		if (e_line[0] == '#' || e_line.size()<1) continue;

		auto vec_record = bgitools::split_str_2_vec(e_line, '\t', 3, 0);
		string &chrname = vec_record[0];

		auto start = (uint64_t)atoi(vec_record[1].c_str());
		auto len = (uint64_t)atoi(vec_record[2].c_str());
		if (len > max_len)
		{
			max_len = len;
		}
		//cout << start << " " << len << endl;

		globalB::chr_start_len id_range;
		id_range.offset = start;
		id_range.len = len;
		globalB::map_chr_range[chrname] = id_range;
	}

	if_vcf_bin_idx.close();
	assert(max_len > 0);
	return globalB::map_chr_range;

}

void globalB::load_chromosome(string & rname, faidx_t *fai)
{
	if (rname[0] == '*')
	{
		return;
	}

	int seq_len;
	char * seq = fai_fetch(fai, rname.c_str(), &seq_len);
	if (seq_len < 0)
	{
		fprintf(stderr, "Failed to fetch sequence in %s\n", rname.c_str());
	}
	globalB::chromosome = seq;
	free(seq);
}
// globalB__ end


//#include "baqtools.cpp"
// baqtools__ start
baqtools::BAQRESULT::BAQRESULT(int sz_bq, int sz_state)
{
	assert(0 != sz_bq);
	assert(0 != sz_state);
	assert(NULL == bq);
	assert(NULL == state);

	bq = new char[sz_bq];
	assert(NULL != bq);

	state = new int[sz_state];
	assert(NULL != state);
}

baqtools::BAQRESULT::~BAQRESULT()
{
	DEL_ARR_MEM(bq);
	DEL_ARR_MEM(state);
}

char baqtools::capBaseByBAQ(char & oq, char & bq, int & state, int & expectedPos)
{
	char b;
	bool isIndel = ((state & 3) != 0); // GATK:stateIsIndel?
	int pos = (state >> 2); // GATK:decode the bit encoded state array values
	if (isIndel || pos != expectedPos) // GATK:we are an indel or we don't align to our best current position
	{
		b = globalB::minBaseQual;
		// GATK:just take b = minBaseQuality
	}
	else
	{
		b = bq < oq ? bq : oq;
	}

	return b;
}

int baqtools::set_u(const int& b, const int& i, const int& k)
{
	int x = i - b;
	x = x > 0 ? x : 0;
	return (k + 1 - x) * 3;
}

void baqtools::hmm_glocal(string & ref, string & query, uint32_t & qstart, uint32_t & l_query, string & _iqual, char *bq, int *state, int &bqSize)
{
	auto allocate_mem_for_s = [](double *p_s, int i_0)
	{
		auto &s = p_s;

		if (NULL != s)
		{
			cout << "p_s should be NULL !" << endl;
			assert(0 == 1);
		}
		s = new double[i_0];
		assert(NULL != s);
		memset(s, 0, sizeof(double) * i_0);
		return s;
	};
	auto allocate_mem_for_f_or_b = [](double **p_p_f, int i_0, int i_1)
	{
		if (NULL != p_p_f)
		{
			cout << "p_p_f should be NULL !" << endl;
			assert(0 == 1);
		}

		auto &f = p_p_f;
		auto &m = i_0;
		auto &n = i_1;
		f = new double*[m];
		for (int i = 0; i < m; i++)
		{
			f[i] = new double[n];
			assert(NULL != f[i]);
		}
		for (int i = 0; i < m; i++)
		{
			memset(f[i], 0, sizeof(double) * n);
		}

		return f;
	};

	auto release_mem_for_f_or_b = [](double **p_p_f, int i_0)
	{
		auto &m = i_0;
		auto &f = p_p_f;
		for (int i = 0; i < i_0; i++)
		{
			DEL_ARR_MEM(f[i]);
		}
		DEL_ARR_MEM(f);
		assert(NULL == f);
	};

	assert(query.length() == _iqual.length());
	assert(l_query >= 1);

	uint32_t l_ref = ref.length(); //GATK: change coordinates

	//GATK: set band width
	uint32_t bw2, bw = l_ref > l_query ? l_ref : l_query;
	if (globalB::cb < abs(int(l_ref - l_query)))
	{
		bw = abs(int(l_ref - l_query)) + 3;
	}

	if (bw > globalB::cb) bw = globalB::cb;
	if (bw < abs(int(l_ref - l_query))) {
		bw = abs(int(l_ref - l_query));
	}
	bw2 = bw * 2 + 1;


	auto i0 = l_query + 1;
	auto i1 = bw2 * 3 + 6;
	auto s_i_0 = l_query + 2;

	double **f = NULL;
	double **b = NULL;
	double *s = NULL;

	f = allocate_mem_for_f_or_b(f, i0, i1);    // mem 0
	b = allocate_mem_for_f_or_b(b, i0, i1);    // mem 1
	s = allocate_mem_for_s(s, s_i_0);           // mem 2

	double *m = baqtools::baqtools_m;

	double sM, sI, bM, bI; // sM, sI = 1/2l
	sM = sI = 1. / (2 * l_query + 2);
	bM = (1 - globalB::cd) / l_ref;
	bI = globalB::cd / l_ref; // (bM+bI)*l_ref==1; bM, bI = (1-alpha)/L; cd = (alpha)gap open probability; ce = (beta)gap extension probability

	m[0 * 3 + 0] = (1 - globalB::cd - globalB::cd) * (1 - sM); m[0 * 3 + 1] = m[0 * 3 + 2] = globalB::cd * (1 - sM);
	m[1 * 3 + 0] = (1 - globalB::ce) * (1 - sI); m[1 * 3 + 1] = globalB::ce * (1 - sI); m[1 * 3 + 2] = 0.;
	m[2 * 3 + 0] = 1 - globalB::ce; m[2 * 3 + 1] = 0.; m[2 * 3 + 2] = globalB::ce;


	/*** forward ***/
	assert(set_u(bw, 0, 0) < i1);
	f[0][set_u(bw, 0, 0)] = s[0] = 1.;

	{ // f[1]
		double *fi = &f[1][0];
		double sum = 0.;
		int beg = 1, end = l_ref < bw + 1 ? l_ref : bw + 1, _beg, _end;
		for (int k = beg; k <= end; ++k)
		{
			int u;
			assert((_iqual[qstart] - 33) >= 0);
			assert((_iqual[qstart] - 33) < globalB::eps_3d_shape[2]);

			double e = globalB::EPSILONS[ref[k - 1]][query[qstart]][(_iqual[qstart] - 33)]; // GATK:calcEpsilon(ref[k-1], query[qstart], _iqual[qstart]);
			u = set_u(bw, 1, k);

			assert(u + 1 < globalB::eps_3d_shape[1]);
			fi[u + 0] = e * bM; fi[u + 1] = globalB::EI * bI;
			sum += fi[u] + fi[u + 1];
		}
		s[1] = sum;
		_beg = set_u(bw, 1, beg); _end = set_u(bw, 1, end); _end += 2;
		for (int k = _beg; k <= _end; ++k)
		{
			fi[k] /= sum;
		}

	}
	for (int i = 2; i <= l_query; ++i)
	{
		double *fi = &f[i][0];
		double *fi1 = &f[i - 1][0];
		double sum = 0.;
		int beg = 1, end = l_ref, x, _beg, _end;
		x = i - bw; beg = beg > x ? beg : x; // band start
		x = i + bw; end = end < x ? end : x; // band end
		for (int k = beg; k <= end; ++k)
		{
			int u, v11, v01, v10;
			assert((_iqual[qstart + i - 1] - 33) < globalB::eps_3d_shape[2]);
			double e = globalB::EPSILONS[ref[k - 1]][query[qstart + i - 1]][(_iqual[qstart + i - 1] - 33)];
			u = set_u(bw, i, k); v11 = set_u(bw, i - 1, k - 1); v10 = set_u(bw, i - 1, k); v01 = set_u(bw, i, k - 1);
			fi[u + 0] = e * (m[0] * fi1[v11 + 0] + m[3] * fi1[v11 + 1] + m[6] * fi1[v11 + 2]);
			fi[u + 1] = globalB::EI * (m[1] * fi1[v10 + 0] + m[4] * fi1[v10 + 1]);
			fi[u + 2] = m[2] * fi[v01 + 0] + m[8] * fi[v01 + 2];
			sum += fi[u] + fi[u + 1] + fi[u + 2];
		}
		s[i] = sum;
		_beg = set_u(bw, i, beg); _end = set_u(bw, i, end); _end += 2;
		sum = 1. / sum;
		for (int k = _beg; k <= _end; ++k) fi[k] *= sum;
	}
	{
		double sum = 0.;
		for (int k = 1; k <= l_ref; ++k)
		{
			int u = set_u(bw, l_query, k);
			if (u < 3 || u >= bw2 * 3 + 3) continue;
			sum += f[l_query][u + 0] * sM + f[l_query][u + 1] * sI;
		}
		s[l_query + 1] = sum; // the last scaling factor
	}

	/*** backward ***/
	for (int k = 1; k <= l_ref; ++k)
	{
		int u = set_u(bw, l_query, k);
		double *bi = &b[l_query][0];
		if (u < 3 || u >= bw2 * 3 + 3) continue;
		bi[u + 0] = sM / s[l_query] / s[l_query + 1]; bi[u + 1] = sI / s[l_query] / s[l_query + 1];
	}
	for (int i = l_query - 1; i >= 1; --i)
	{
		int beg = 1, end = l_ref, x, _beg, _end;
		double *bi = &b[i][0];
		double *bi1 = &b[i + 1][0];
		double y = (i > 1) ? 1. : 0.;
		x = i - bw; beg = beg > x ? beg : x;
		x = i + bw; end = end < x ? end : x;
		for (int k = end; k >= beg; --k)
		{
			int u, v11, v01, v10;
			u = set_u(bw, i, k); v11 = set_u(bw, i + 1, k + 1); v10 = set_u(bw, i + 1, k); v01 = set_u(bw, i, k + 1);
			double e = (k >= l_ref ? 0 : globalB::EPSILONS[ref[k]][query[qstart + i]][(_iqual[qstart + i] - 33)]) * bi1[v11];
			bi[u + 0] = e * m[0] + globalB::EI * m[1] * bi1[v10 + 1] + m[2] * bi[v01 + 2]; // bi1[v11] has been foled into e.
			bi[u + 1] = e * m[3] + globalB::EI * m[4] * bi1[v10 + 1];
			bi[u + 2] = (e * m[6] + m[8] * bi[v01 + 2]) * y;
		}
		// rescale
		_beg = set_u(bw, i, beg); _end = set_u(bw, i, end); _end += 2;
		y = 1. / s[i];
		for (int k = _beg; k <= _end; ++k) bi[k] *= y;
	}

	double pb;
	{ // b[0]
		int beg = 1, end = l_ref < bw + 1 ? l_ref : bw + 1;
		double sum = 0.;
		for (int k = end; k >= beg; --k) {
			int u = set_u(bw, 1, k);
			double e = globalB::EPSILONS[ref[k - 1]][query[qstart]][(_iqual[qstart] - 33)];
			if (u < 3 || u >= bw2 * 3 + 3) continue;
			sum += e * b[1][u + 0] * bM + globalB::EI * b[1][u + 1] * bI;
		}
		pb = b[0][set_u(bw, 0, 0)] = sum / s[0]; // if everything works as is expected, pb == 1.0
		if (fabs(pb - 1.0) >= 10e-4)
		{
			assert(fabs(pb - 1.0) >= 10e-4);
		}
	}


	/*** MAP ***/
	for (int i = 1; i <= l_query; ++i)
	{
		double sum = 0., max = 0.;
		double *fi = &f[i][0];
		double *bi = &b[i][0];
		int beg = 1, end = l_ref, x, max_k = -1;
		x = i - bw; beg = beg > x ? beg : x;
		x = i + bw; end = end < x ? end : x;
		for (int k = beg; k <= end; ++k) {
			int u = set_u(bw, i, k);
			double z;
			sum += (z = fi[u + 0] * bi[u + 0]); if (z > max) { max = z; max_k = (k - 1) << 2 | 0; }
			sum += (z = fi[u + 1] * bi[u + 1]); if (z > max) { max = z; max_k = (k - 1) << 2 | 1; }
		}
		max /= sum; sum *= s[i]; // if everything works as is expected, sum == 1.0
		if (fabs(sum - 1.0) >= 10e-4)
		{
			assert(fabs(sum - 1.0) >= 10e-4);
		}
		if (sizeof(state) != 0) state[qstart + i - 1] = max_k;
		if (bqSize != 0)
		{
			int k = (int)(-4.343 * log(1. - max) + .499); // = 10*log10(1-max)
			bq[qstart + i - 1] = (k > 100 ? 99 : (k < globalB::minBaseQual ? globalB::minBaseQual : k));
		}
		// GATK:System.out.println("("+pb+","+sum+")"+" ("+(i-1)+","+(max_k>>2)+","+(max_k&3)+","+max+")");
	}

	DEL_ARR_MEM(s); // d mem 2
	release_mem_for_f_or_b(b, i0); // d mem 1
	release_mem_for_f_or_b(f, i0); // d mem 0
}

string baqtools::calcBAQFromHMM(sam_record& e_sam_record, string & refSeq)
{
	uint32_t readStart = e_sam_record.pos;

	uint32_t offset = globalB::cb / 2; //ensure offset = 3!!!
	assert(offset == 3);

	uint32_t start = max(int(readStart - offset - samtools::getInsertionOffset(e_sam_record, 0)), 0);
	//hmm------------------------------------
	int refOffset = (int)(start - readStart);
	pair<int, int> queryRange = samtools::calculateQueryRange(e_sam_record); // return the start end pair, if 50M, then <0,50>

	uint32_t queryStart = queryRange.first;
	uint32_t queryEnd = queryRange.second;
	assert(queryStart >= 0);
	assert(queryEnd >= 0);
	assert(queryEnd >= queryStart);

	// GATK: note -- assumes ref is offset from the *CLIPPED* start
	string quals = e_sam_record.qual;
	string query = e_sam_record.seq;
	uint32_t queryLen = queryEnd - queryStart;

	int bqsize = quals.length();
	baqtools::BAQRESULT baqResult_(bqsize, bqsize);  // alloc a baqResult
	auto *baqResult = &baqResult_;

	assert(NULL != baqResult->bq);
	assert(NULL != baqResult->state);

	baqResult->refBases = refSeq;
	baqResult->readBases = query;
	baqResult->queryStart = queryStart;
	baqResult->queryLen = queryLen;
	baqResult->rawQuals = quals;

	baqtools::hmm_glocal(
			baqResult->refBases,		// ref
			baqResult->readBases,       // query
			baqResult->queryStart,		// qstart
			baqResult->queryLen,		// l_query
			baqResult->rawQuals,		// _iqual
			baqResult->bq,				// *bq
			baqResult->state,			// *state
			bqsize						// bqsize
			); //890ms

	uint32_t readI = 0, refI = 0;
	unordered_map<int, struct bgitools::CIGAR> cigar_map = bgitools::get_cigar(e_sam_record);
	for (int iii = 0; iii < cigar_map.size(); iii++)
	{
		struct bgitools::CIGAR elt = cigar_map[iii];
		uint32_t l = elt.length;
		switch (elt.opt) {
			case bgitools::OPT::N: // cannot handle these
				return NULL;
			case bgitools::OPT::H: case bgitools::OPT::P: // ignore pads and hard clips
				break;
			case bgitools::OPT::S: refI += l; // move the reference too, in addition to I
			case bgitools::OPT::I:
					       // GATK:todo -- is it really the case that we want to treat I and S the same?
					       for (int i = readI; i < readI + l; i++)
					       {
						       baqResult->bq[i] = baqResult->rawQuals[i] - 33;
					       }
					       readI += l;
					       break;
			case bgitools::OPT::D: refI += l; break;
			case bgitools::OPT::M:
					       for (uint32_t i = readI; i < readI + l; i++) {
						       int expectedPos = refI - refOffset + (i - readI);
						       char oq = (baqResult->rawQuals[i] - 33);
						       char bq = baqResult->bq[i];
						       int state = (baqResult->state[i]);
						       baqResult->bq[i] = baqtools::capBaseByBAQ(oq, bq, state, expectedPos);
					       }
					       readI += l; refI += l;
					       break;
			default:
					       cout << "BUG: Unexpected CIGAR element in read " << e_sam_record.qname << endl;
					       assert(0 == 1);
		}
	}

	if (readI != bqsize) // odd cigar string
	{
		// GATK:System.arraycopy(baqResult.rawQuals, 0, baqResult.bq, 0, baqResult.bq.length);
		string _bq = string(baqResult->rawQuals.begin(), baqResult->rawQuals.begin() + bqsize);
		for (auto & ch : _bq)
		{
			ch -= 33;
		}
		return _bq;

	}

	// TODO: replace it by effcient method
	return string(baqResult->bq, baqResult->bq + bqsize);
}

string baqtools::getBAQTag(sam_record& e_sam_record, string & refSeq)
{
	string ret_str = string("");
	string quals = e_sam_record.qual;
	string bq_hmmResult = baqtools::calcBAQFromHMM(e_sam_record, refSeq);  // return some new raw bytes  //900ms

	int bqLen = bq_hmmResult.length();
	char *bqTag = new char[bqLen]; // mem 0
	assert(NULL != bqTag);

	if (bqLen != 0)
	{
		// GATK:Offset to base alignment quality (BAQ), of the same length as the read sequence.
		// At the i-th read base, BAQi = Qi - (BQi - 64) where Qi is the i-th base quality.
		// so BQi = Qi - BAQi + 64

		for (int i = 0; i < bqLen; i++)
		{
			int bq = quals[i] - 33 + 64;
			int baq_i = (int)bq_hmmResult[i];
			int tag = bq - baq_i;
			if (tag < 0)
			{
				cout << "BAQ tag calculation error.  BAQ value above base quality at " << e_sam_record.qname << endl;
				assert(0 == 1);
			}
			if (tag > globalB::MAX_VALUE)
			{
				cout << "we encountered an extremely high quality score at " << e_sam_record.qname << endl;
				assert(0 == 1);
			}
			bqTag[i] = tag;
		}
		ret_str = string(bqTag, bqTag + bqLen);
	}


	DEL_ARR_MEM(bqTag); // d mem 0
	return ret_str;
}

// baqtools__ end


//#include "sam_record.cpp"
// sam_record__ start
string sam_record::header_str = string("initial_sam_header_str");
string sam_record::group_name = string("initial_sam_group_name");
string sam_record::BI = string("initial_sam_BI");
string sam_record::BD = string("initial_sam_BD");
const int sam_record::VEC_SZ = 11;

sam_record::sam_record(const string & e_line)
{
	vector<string> vec_line = bgitools::split_str_2_vec(e_line, '\t', sam_record::VEC_SZ, 1);
	assert(vec_line.size() == 12);

	qname = vec_line[0];
	flag = (uint16_t)atoi(vec_line[1].c_str());
	rname = vec_line[2];
	pos = (uint32_t)atoi(vec_line[3].c_str());

	mapq = (uint8_t)atoi(vec_line[4].c_str());
	cigar = vec_line[5];
	rnext = vec_line[6];
	pnext = (uint32_t)atoi(vec_line[7].c_str());

	tlen = (int32_t)atoi(vec_line[8].c_str());
	seq = vec_line[9];
	qual = vec_line[10];
	ex_str = vec_line[sam_record::VEC_SZ];
}

string sam_record::to_string()
{

	const int MAX_CHAR_E_LINE = 1024 * 2;

	char e_line[MAX_CHAR_E_LINE] = { 0 };

	const char *record_format =
		"%s\t%u\t%s\t%u\t%u\t%s\t%s\t%u\t%0d\t%s\t%s\t%s\n";

	snprintf(e_line, MAX_CHAR_E_LINE, record_format,
			qname.c_str(), flag, rname.c_str(), pos, mapq,  // 5 items
			cigar.c_str(), rnext.c_str(), pnext, tlen, seq.c_str(), //5 items
			qual.c_str(), ex_str_to_string().c_str() // 2 items
		);

	return e_line;
}

string& sam_record::ex_str_to_string()
{

	int pos_bd = 0;
	int pos_bi = 0;
	auto bin_num_of_ext = [](string & e_ext)
	{
		int ret_num = (e_ext[1] << 8) | e_ext[0];
		return ret_num;
	};


	auto vec_ex_str = bgitools::split_str_2_vec(ex_str, '\t');

	size_t i = 0;

	auto sz = vec_ex_str.size();
	for (i = 0; i<sz; i++)
	{
		if (bin_num_of_ext(BD) < bin_num_of_ext(vec_ex_str[i]))
		{
			break;
		}
	}
	pos_bd = i;
	auto c_it_bd = vec_ex_str.cbegin();
	vec_ex_str.insert(c_it_bd + pos_bd, BD);

	sz = vec_ex_str.size();
	for (i = pos_bd; i<sz; i++)
	{
		if (bin_num_of_ext(BI) < bin_num_of_ext(vec_ex_str[i]))
		{
			break;
		}
	}
	pos_bi = i; // because we already insert our BD
	auto c_it_bi = vec_ex_str.cbegin();
	vec_ex_str.insert(c_it_bi + pos_bi, BI);

	//string ret_str = bgitools::join_vec_2_str(vec_ex_str, '\t');
	ex_str = bgitools::join_vec_2_str(vec_ex_str, '\t');
	return ex_str;
}

// sam_record__ end



//#include "grp_tables.cpp"
// grp_tables__ start
vector<struct grptools::QualInterval> grp_tables::quantize(vector<long> & qualHistogram)
{
	vector<struct grptools::QualInterval> intervals;
	for (int qStart = 0; qStart < qualHistogram.size(); qStart++)
	{
		long nObs = qualHistogram[qStart];
		double errorRate = globalB::qualToErrorProbCache[qStart & 0xff];
		double nErrors = nObs * errorRate;

		struct grptools::QualInterval qi;
		qi.qStart = qStart;
		qi.qEnd = qStart;
		qi.nObservations = nObs;
		qi.nErrors = (int)floor(nErrors);
		qi.fixedQual = qStart;
		qi.level = 0;
		qi.mergeOrder = 0;

		intervals.push_back(qi);
	}

	while (intervals.size() > globalB::QUANTIZING_LEVELS)
	{
		vector<struct grptools::QualInterval>::const_iterator cit1 = intervals.cbegin();
		vector<struct grptools::QualInterval>::const_iterator cit1p = intervals.cbegin();
		cit1p++;

		struct grptools::QualInterval minMerge;
		int lastMergeOrder = 0;
		while (cit1p != intervals.cend())
		{
			struct grptools::QualInterval left = *cit1++;
			struct grptools::QualInterval right = *cit1p++;

			struct grptools::QualInterval merged = grptools::merge(left, right);
			lastMergeOrder = max(max(lastMergeOrder, left.mergeOrder), right.mergeOrder);

			if (minMerge.qStart == -1)
			{
				minMerge = merged;
				continue;
			}

			double mergedGlobalErrorRate = grptools::getErrorRate(merged.nObservations, merged.nErrors, merged.fixedQual);
			double minMergedGlobalErrorRate = grptools::getErrorRate(minMerge.nObservations, minMerge.nErrors, minMerge.fixedQual);
			if (grptools::getPenalty(merged, mergedGlobalErrorRate) < grptools::getPenalty(minMerge, minMergedGlobalErrorRate))
			{
				minMerge = merged;
			}
		}

		intervals = grptools::removeAndAdd(intervals, minMerge);
		minMerge.mergeOrder = lastMergeOrder + 1;

	}

	return intervals;

}

void grp_tables::Quantized_option()
{
	vector<long> qualHistogram(globalB::MAX_PHRED_SCORE + 1, 0); // GATK:create a histogram with the empirical quality distribution

	for (auto & it : RecalTable1)
	{
		if (fabs(it.observations) >= 10e-4)
		{
			it.empiricalQuality = grptools::getEmpiricalQuality(it.empiricalQuality,
					it.observations,
					it.numMismatches);
			int empiricalQual = (int)round(it.empiricalQuality);
			qualHistogram[empiricalQual] += (long)it.observations;
		}
	}

	vector<struct grptools::QualInterval> intervals = quantize(qualHistogram);
	for (int i = 0; i < intervals.size(); i++)
	{
		grptools::QualInterval interval = intervals[i];
		for (int q = interval.qStart; q <= interval.qEnd; q++)
		{
			quantized[q].count = qualHistogram[q];
			if (interval.fixedQual == -1)
			{
				long double errorRate = grptools::getErrorRate(interval.nObservations, interval.nErrors, interval.fixedQual);
				long double eps = 0.;
				long double prob = 1 - errorRate;
				quantized[q].quantizedScore = grptools::probToQual(prob, eps);
			}
			else
			{
				quantized[q].quantizedScore = interval.fixedQual;
			}
		}
	}
}

void grp_tables::recalTable0_option(int & key, double & isError)
{
	double existedErrors = grptools::calcExpectedErrors(recalTable0.estimatedQReported, recalTable0.observations);
	double newErrors = grptools::calcExpectedErrors((double)key, 1.0);
	double sumErrors = existedErrors + newErrors;
	recalTable0.observations += 1.;
	recalTable0.numMismatches += isError;
	recalTable0.estimatedQReported = -10 * log10(sumErrors / recalTable0.observations);
}

void grp_tables::RecalTable1_option(int & key, double & isError)
{
	RecalTable1[key].estimatedQReported = (double)key;
	RecalTable1[key].observations += 1.;
	RecalTable1[key].numMismatches += isError;
}

string grp_tables::arguments_to_string()
{
	string sb("");
	sb += arguments.to_string();
	return sb;
}

void grp_tables::allocateRT2()
{
	RecalTable2 = new grptools::RecalTable2**[globalB::GrpRT2SHAPE_dim_0];
	assert(NULL != RecalTable2);
	for (int i = 0; i < globalB::GrpRT2SHAPE_dim_0; i++)
	{
		RecalTable2[i] = NULL;
		RecalTable2[i] = new grptools::RecalTable2*[globalB::GrpRT2SHAPE_dim_1];
		assert(NULL != RecalTable2[i]);
		for (int j = 0; j < globalB::GrpRT2SHAPE_dim_1; j++)
		{
			RecalTable2[i][j] = NULL;
			RecalTable2[i][j] = new grptools::RecalTable2[globalB::GrpRT2SHAPE_dim_2];
			assert(NULL != RecalTable2[i][j]);
		}
	}
}

void grp_tables::releaseRT2()
{
	for (int i = 0; i < globalB::GrpRT2SHAPE_dim_0; i++)
	{
		for (int j = 0; j < globalB::GrpRT2SHAPE_dim_1; j++)
		{
			DEL_ARR_MEM(RecalTable2[i][j]);
		}

		DEL_ARR_MEM(RecalTable2[i]);
	}

	DEL_ARR_MEM(RecalTable2);
}

void grp_tables::RecalTable2_option(grptools::CVKEY & key, double & isError, int & covariateIndex)
{
	int cvKey = covariateIndex == 0 ? key.contextKey : key.cycleKey;
	RecalTable2[covariateIndex][key.quality][cvKey].observations += 1.;
	RecalTable2[covariateIndex][key.quality][cvKey].numMismatches += isError;
}

string grp_tables::arguments_fcout(const string & fn)
{
	string ret_str = arguments_to_string();
	bgitools::fcout(fn, ret_str);
	return ret_str;
}

string grp_tables::quantized_to_string()
{
	const char *comment_format =
		"#:GATKTable:3:%d:%%s:%%s:%%s:;\n"
		"#:GATKTable:Quantized:Quality quantization map\n";

	const char *record_format = "%12s%7s%16s\n";
	auto *comment = e_line;
	snprintf(comment, MAX_CHAR_E_LINE, comment_format, quantized.size());

	string sb(comment);

	snprintf(e_line, MAX_CHAR_E_LINE, record_format, "QualityScore", "Count", "QuantizedScore");
	sb += e_line;

	vector<long> vec_q_key = bgitools::extract_keys(quantized);
	sort(vec_q_key.begin(), vec_q_key.end());

	for (auto & qk : vec_q_key)
	{
		sb += quantized[qk].to_string(qk, record_format, e_line, grp_tables::MAX_CHAR_E_LINE);
	}

	return sb;
}


string grp_tables::quantized_fcout(const string & fn)
{
	string ret_str = quantized_to_string();
	bgitools::fcout(fn, ret_str);
	return ret_str;
}


string grp_tables::recalTable0_to_string()
{
	const char *comment_format =
		"#:GATKTable:6:%d:%%s:%%s:%%.4f:%%.4f:%%.2f:%%.2f:;\n"
		"#:GATKTable:RecalTable0:\n";

	auto *comment = e_line;
	snprintf(comment, MAX_CHAR_E_LINE, comment_format, 1);
	string sb(comment);

	const char *field_format = "%-20s%-11s%-18s%-20s%-14s%-6s\n";
	auto *field = e_line;
	snprintf(field, MAX_CHAR_E_LINE, field_format, "ReadGroup", "EventType", "EmpiricalQuality", "EstimatedQReported", "Observations", "Errors");
	sb += field;


	const char *record_format = "%-20s%-11s%16.4f%20.4f%14.2f%14.2f\n";
	string ReadGroup(sam_record::group_name);
	bgitools::trim_str(ReadGroup);
	string EventType("M");

	sb += recalTable0.to_string(ReadGroup, EventType, record_format, e_line, MAX_CHAR_E_LINE);

	return sb;
}

string grp_tables::recalTable0_fcout(const string & fn)
{
	string ret_str = recalTable0_to_string();
	bgitools::fcout(fn, ret_str);
	return ret_str;
}

string grp_tables::RecalTable1_to_string()
{
	const char *comment_format =
		"#:GATKTable:6:%d:%%s:%%s:%%s:%%.4f:%%.2f:%%.2f:;\n"
		"#:GATKTable:RecalTable1:\n";

	vector<int> qual_vec;
	for (int i = 0; i < 100; i++)
	{
		if (RecalTable1[i].observations > 10e-4)
		{
			qual_vec.push_back(i);
		}
	}
	auto *comment = e_line;
	snprintf(comment, MAX_CHAR_E_LINE, comment_format, qual_vec.size());
	string sb(comment);

	const char *field_format = "%-20s%-14s%-11s%-19s%-14s%-6s\n";
	auto *field = e_line;
	snprintf(field, MAX_CHAR_E_LINE, field_format, "ReadGroup", "QualityScore", "EventType", "EmpiricalQuality", "Observations", "Errors");
	sb += field;

	const char *record_format = "%-20s%-14d%-11s%-14.4f%-12.2f%-4.2f\n";

	string ReadGroup(sam_record::group_name);
	bgitools::trim_str(ReadGroup);
	string EventType("M");

	for (auto &key : qual_vec)
	{
		sb += RecalTable1[key].to_string(ReadGroup, key, EventType, record_format, e_line, MAX_CHAR_E_LINE);
	}

	return sb;
}

string grp_tables::RecalTable2_to_string()
{
	const char *comment_format =
		"#:GATKTable:8:%d:%%s:%%s:%%s:%%s:%%s:%%.4f:%%.2f:%%.2f:;\n"
		"#:GATKTable:RecalTable2:\n";

	vector<struct grptools::RT2KEY> bitKey;
	struct grptools::RT2KEY id_RT2KEY;
	for (int i = 0; i < 2; i++)// covariateIdx
	{
		for (int ii = 0; ii < 256; ii++) //quality
		{
			for (int iii = 0; iii < 256; iii++) //covariateKey
			{
				if (fabs(RecalTable2[i][ii][iii].observations) > 10e-4)
				{
					id_RT2KEY.covariateIdx = i;
					id_RT2KEY.quality = ii;
					id_RT2KEY.covariateKey = iii;
					bitKey.push_back(id_RT2KEY);
				}
			}
		}
	}
	auto *comment = e_line;
	snprintf(comment, MAX_CHAR_E_LINE, comment_format, bitKey.size());
	string sb(comment);

	const char *field_format = "%-20s%-14s%-16s%-15s%-11s%-18s%-14s%-6s\n";
	auto *field = e_line;
	snprintf(field, MAX_CHAR_E_LINE, field_format, "ReadGroup", "QualityScore", "CovariateValue", "CovariateName", "EventType", "EmpiricalQuality", "Observations", "Errors");
	sb += field;

	const char *record_format = "%-20s%-14s%-16s%-15s%-11s%-14.4f%-12.2f%-4.2f\n";

	string ReadGroup(sam_record::group_name);
	bgitools::trim_str(ReadGroup);
	string EventType("M");

	for (auto & key : bitKey)
	{
		sb += RecalTable2[key.covariateIdx][key.quality][key.covariateKey].to_string(ReadGroup, key, EventType, record_format, e_line, MAX_CHAR_E_LINE); // ls_
	}

	return sb;
}

string grp_tables::grp_tables_fcout(const string & fn)
{
	string sb =
		arguments_to_string() + "\n" +
		quantized_to_string() + "\n" +
		recalTable0_to_string() + "\n" +
		RecalTable1_to_string() + "\n" +
		RecalTable2_to_string();

	bgitools::fcout(fn, sb);
	return sb;
}

void grp_tables::updateDataForRead(unordered_map<int, struct grptools::CVKEY> & readCovariates, vector<bool> & skip, vector<double> & snpErrors)
{
	std::lock_guard<std::mutex> lock(globalB::g_mutex);
	int skipLen = skip.size();
	struct grptools::CVKEY cvkey;
	int covariateIndex;
	double isError;
	for (int offset = 0; offset < skipLen; offset++)
	{
		if (!skip[offset])
		{
			isError = snpErrors[offset];

			cvkey = readCovariates[offset];
			// TODO:final int eventIndex = EventType.BASE_SUBSTITUTION.index;

			recalTable0_option(cvkey.quality, isError);
			RecalTable1_option(cvkey.quality, isError);

			if (cvkey.contextKey >= 0)
			{
				covariateIndex = 0;
				RecalTable2_option(cvkey, isError, covariateIndex);
			}
			if (cvkey.cycleKey >= 0)
			{
				covariateIndex = 1;
				RecalTable2_option(cvkey, isError, covariateIndex);
			}
		}
	}
}

void grp_tables::updateDataForRead_v1(vector<struct grptools::CVKEY> & readCovariates, vector<bool> & skip, vector<double> & snpErrors)
{
	std::lock_guard<std::mutex> lock(globalB::g_mutex);
	int skipLen = skip.size();
	struct grptools::CVKEY cvkey;
	int covariateIndex;
	double isError;
	for (int offset = 0; offset < skipLen; offset++)
	{
		if (!skip[offset])
		{
			isError = snpErrors[offset];

			cvkey = readCovariates[offset];
			// TODO:final int eventIndex = EventType.BASE_SUBSTITUTION.index;

			recalTable0_option(cvkey.quality, isError);
			RecalTable1_option(cvkey.quality, isError);

			if (cvkey.contextKey >= 0)
			{
				covariateIndex = 0;
				RecalTable2_option(cvkey, isError, covariateIndex);
			}
			if (cvkey.cycleKey >= 0)
			{
				covariateIndex = 1;
				RecalTable2_option(cvkey, isError, covariateIndex);
			}
		}
	}
}

void grp_tables::updateDataForRead_v2(grptools::grpTempDat * grpTempDatArr, int & size)
{
	std::lock_guard<std::mutex> lock(globalB::g_mutex);
	for (int i = 0; i < size; i++)
	{
		int skipLen = grpTempDatArr[i].length;
		if (skipLen != 0)
		{
			struct grptools::CVKEY cvkey;
			int covariateIndex;
			double isError;
			grptools::grpTempDat *grpTempDatStruct = &grpTempDatArr[i];

			for (int offset = 0; offset < skipLen; offset++)
			{
				if (!(*grpTempDatStruct).skip[offset])
				{
					isError = (*grpTempDatStruct).snpErrors[offset];

					cvkey = (*grpTempDatStruct).readCovariates[offset];
					// TODO:final int eventIndex = EventType.BASE_SUBSTITUTION.index;

					recalTable0_option(cvkey.quality, isError);
					RecalTable1_option(cvkey.quality, isError);

					if (cvkey.contextKey >= 0)
					{
						covariateIndex = 0;
						RecalTable2_option(cvkey, isError, covariateIndex);
					}
					if (cvkey.cycleKey >= 0)
					{
						covariateIndex = 1;
						RecalTable2_option(cvkey, isError, covariateIndex);
					}
				}
			}

		}
	}

	for (int i = 0; i < size; i++)
	{
		if (grpTempDatArr[i].length != 0)
		{
			free(grpTempDatArr[i].readCovariates);
			free(grpTempDatArr[i].snpErrors);
			free(grpTempDatArr[i].skip);
		}
	}
	free(grpTempDatArr);
}

void grp_tables::updateDataForRead_v3(grptools::grpTempDat * grpTempDatArr, int & size)
{
	for (int i = 0; i < size; i++)
	{
		int skipLen = grpTempDatArr[i].length;
		if (skipLen != 0)
		{
			struct grptools::CVKEY cvkey;
			int covariateIndex;
			double isError;
			grptools::grpTempDat *grpTempDatStruct = &grpTempDatArr[i];

			for (int offset = 0; offset < skipLen; offset++)
			{
				if (!(*grpTempDatStruct).skip[offset])
				{
					isError = (*grpTempDatStruct).snpErrors[offset];

					cvkey = (*grpTempDatStruct).readCovariates[offset];
					// TODO:final int eventIndex = EventType.BASE_SUBSTITUTION.index;

					RecalTable1_option(cvkey.quality, isError);

					if (cvkey.contextKey >= 0)
					{
						covariateIndex = 0;
						RecalTable2_option(cvkey, isError, covariateIndex);
					}
					covariateIndex = 1;
					RecalTable2_option(cvkey, isError, covariateIndex);
				}
			}

		}
	}

	for (int i = 0; i < size; i++)
	{
		if (grpTempDatArr[i].length != 0)
		{
			free(grpTempDatArr[i].readCovariates);
			free(grpTempDatArr[i].snpErrors);
			free(grpTempDatArr[i].skip);
		}
	}
	free(grpTempDatArr);
}


void grp_tables::load(string &grp_path)
{
	const int MAX_CHAR = 1024;
	auto F_R = (ios::in);
	ifstream if_(grp_path.c_str(), F_R);  assert(if_.is_open());

	char line_content[MAX_CHAR];
	line_content[MAX_CHAR - 1] = '\0';
	if_.seekg(0, ios::beg);
	while (!if_.eof())
	{
		if_.getline(line_content, MAX_CHAR);    // don't need read too long
		if (string(line_content) == "#:GATKTable:RecalTable0:")
		{
			if_.getline(line_content, MAX_CHAR);
			if_.getline(line_content, MAX_CHAR);
			bgitools::trim_str(line_content);
			if (strlen(line_content) != 0)
			{
				continue;
			}
			else
			{
				string rt0Dat = line_content;
				vector<string> vec_rt0 = bgitools::split_str_2_vec(rt0Dat, ' ');
				recalTable0.empiricalQuality = std::stold(vec_rt0[2]);
				recalTable0.estimatedQReported = std::stold(vec_rt0[3]);
				recalTable0.observations = std::stold(vec_rt0[4]);
				recalTable0.numMismatches = std::stold(vec_rt0[5]);
			}
		}
		if (string(line_content) == "#:GATKTable:RecalTable1:")
		{
			if_.getline(line_content, MAX_CHAR);
			if_.getline(line_content, MAX_CHAR);
			bgitools::trim_str(line_content);
			while (!if_.eof() && strlen(line_content) != 0)
			{
				string rt1Dat = line_content;
				vector<string> vec_rt1 = bgitools::split_str_2_vec(rt1Dat, ' ');
				int qualityScore = (int)stol(vec_rt1[1].c_str());
				RecalTable1[qualityScore].empiricalQuality = std::stold(vec_rt1[3]);
				RecalTable1[qualityScore].observations = std::stold(vec_rt1[4]);
				RecalTable1[qualityScore].numMismatches = std::stold(vec_rt1[5]);
				if_.getline(line_content, MAX_CHAR);
				bgitools::trim_str(line_content);
			}
		}
		if (string(line_content) == "#:GATKTable:RecalTable2:")
		{
			if_.getline(line_content, MAX_CHAR);
			if_.getline(line_content, MAX_CHAR);
			bgitools::trim_str(line_content);
			while (!if_.eof() && strlen(line_content) != 0)
			{
				string rt1Dat = line_content;
				vector<string> vec_rt1 = bgitools::split_str_2_vec(rt1Dat, ' ');
				int qualityScore = (int)stol(vec_rt1[1].c_str());
				string covariateValue = vec_rt1[2];
				int key, cvInd;
				if (vec_rt1[3] == "Context")
				{
					cvInd = 0;
					key = samtools::keyFromContext(covariateValue, 0, globalB::mismatchesContextSize);
				}
				else
				{
					cvInd = 1;
					int cycle = atoi(covariateValue.c_str());
					key = samtools::keyFromCycle(cycle);
				}
				RecalTable2[cvInd][qualityScore][key].empiricalQuality = std::stold(vec_rt1[5]);
				RecalTable2[cvInd][qualityScore][key].observations = std::stold(vec_rt1[6]);
				RecalTable2[cvInd][qualityScore][key].numMismatches = std::stold(vec_rt1[7]);
				if_.getline(line_content, MAX_CHAR);
				bgitools::trim_str(line_content);
			}
		}
		continue;
	}
	if_.close();
}

// grp_tables__ end



//#include "bqsrtools.cpp"
// bqsrtools__ start
void bqsrtools::grp_reduce(grp_tables & gt, grp_tables * grp_table_arr, int & td_num)
{
	for (int i = 0; i < td_num; i++)
	{
		// recalTable1 reduce
		for (int qual = 0; qual < globalB::GrpRT1LENGTH; qual++)
		{
			if (grp_table_arr[i].RecalTable1[qual].observations > 10e-4)
			{
				gt.RecalTable1[qual].estimatedQReported = (double)qual;
				gt.RecalTable1[qual].observations += grp_table_arr[i].RecalTable1[qual].observations;
				gt.RecalTable1[qual].numMismatches += grp_table_arr[i].RecalTable1[qual].numMismatches;
			}
		}

		//recalTable2 reduce
		for (int cvInd = 0; cvInd < globalB::GrpRT2SHAPE_dim_0; cvInd++)
		{
			for (int qual = 0; qual < globalB::GrpRT2SHAPE_dim_1; qual++)
			{
				for (int cvKey = 0; cvKey < globalB::GrpRT2SHAPE_dim_2; cvKey++)
				{
					if (grp_table_arr[i].RecalTable2[cvInd][qual][cvKey].observations > 10e-4)
					{
						gt.RecalTable2[cvInd][qual][cvKey].observations += grp_table_arr[i].RecalTable2[cvInd][qual][cvKey].observations;
						gt.RecalTable2[cvInd][qual][cvKey].numMismatches += grp_table_arr[i].RecalTable2[cvInd][qual][cvKey].numMismatches;
					}
				}
			}
		}
	}

	// recalTable0 reduce
	double eQR = 0;
	for (int q = 0; q < globalB::GrpRT1LENGTH; q++)
	{
		if (gt.RecalTable1[q].observations > 10e-4)
		{
			gt.recalTable0.observations += gt.RecalTable1[q].observations;
			gt.recalTable0.numMismatches += gt.RecalTable1[q].numMismatches;
			//This equation can be easily proved.
			eQR += (globalB::convertFromPhredScale(gt.RecalTable1[q].estimatedQReported) * gt.RecalTable1[q].observations);
		}
	}
	gt.recalTable0.estimatedQReported = -10.0*log10(eQR / gt.recalTable0.observations);

	globalB::grp_release(td_num);
}

double bqsrtools::CalculateDeltaQCovariates(struct grptools::RecalTable2 *** frt2, vector<int> & keyVec, double & globalDeltaQ, double & deltaQReported)
{
	double result = 0.0;

	// GATK:for all optional covariates
	for (int i = 0; i < 2; i++) {
		if (keyVec[i + 1] < 0)
			continue;

		if (fabs(frt2[i][keyVec[0]][keyVec[i + 1]].observations) >= 10e-4)
		{
			double deltaQCovariateEmpirical = grptools::getEmpiricalQuality(frt2[i][keyVec[0]][keyVec[i + 1]].empiricalQuality,
					frt2[i][keyVec[0]][keyVec[i + 1]].observations,
					frt2[i][keyVec[0]][keyVec[i + 1]].numMismatches);
			result += (deltaQCovariateEmpirical - keyVec[0] - (globalDeltaQ + deltaQReported));
		}
	}
	return result;
}

double bqsrtools::CalculateDeltaQReported(struct grptools::RecalTable1 * frt1, double & globalDeltaQ, int & qualFromRead)
{
	double result = 0.0;

	if (frt1[qualFromRead].observations > 10e-4)
	{
		double deltaQReportedEmpirical = grptools::getEmpiricalQuality(frt1[qualFromRead].empiricalQuality,
				frt1[qualFromRead].observations,
				frt1[qualFromRead].numMismatches);
		result = deltaQReportedEmpirical - qualFromRead - globalDeltaQ;
	}

	return result;
}

double bqsrtools::calculateGlobalDeltaQ(struct grptools::RecalTable0 & rt0)
{
	double result = 0.0;
	if (rt0.observations > 10e-4)
	{
		double globalDeltaQEmpirical = grptools::getEmpiricalQuality(rt0.empiricalQuality, rt0.observations, rt0.numMismatches);
		double aggregrateQReported = rt0.estimatedQReported;
		result = globalDeltaQEmpirical - aggregrateQReported;
	}

	return result;
}

int bqsrtools::PerformSequentialQualityCalculation(grp_tables & gt, vector<int> & keySet, bgitools::EventType & errorModel)
{
	//todo:some day implement it with errorModel, now leave it alone
	double globalDeltaQ = bqsrtools::calculateGlobalDeltaQ(gt.recalTable0);
	double deltaQReported = bqsrtools::CalculateDeltaQReported(gt.RecalTable1, globalDeltaQ, keySet[0]);
	double deltaQCovariates = bqsrtools::CalculateDeltaQCovariates(gt.RecalTable2, keySet, globalDeltaQ, deltaQReported);

	// GATK:calculate the recalibrated qual using the BQSR formula
	double recalibratedQual = keySet[0] + globalDeltaQ + deltaQReported + deltaQCovariates;
	// GATK:recalibrated quality is bound between 1 and MAX_QUAL
	int rq = (recalibratedQual > 0.0) ? (int)(recalibratedQual + 0.5001) : (int)(recalibratedQual - 0.5001);
	//    int rq = round(recalibratedQual);
	recalibratedQual = max(min(rq, globalB::MAX_PHRED_SCORE), 1);

	return recalibratedQual; // quantized table has no use in printReads!
}
// bqsrtools__ end




#if 0
// ****** headers, ordered *****
#include "bgitools.hpp"
#include "globalb.hpp"
#include "samtools.hpp"
#include "grptools.hpp"
#include "bqsrtools.hpp"
#include "baqtools.hpp"
#include "sam_record.hpp"
#include "bed_record.hpp"
#include "grp_tables.hpp"


// ****** cpps, unordered ******
#include "bed_record.cpp"
#include "bgitools.cpp"
#include "samtools.cpp"
#include "grptools.cpp"
#include "globalb.cpp"
#include "baqtools.cpp"
#include "sam_record.cpp"
#include "grp_tables.cpp"
#include "bqsrtools.cpp"

#endif

#if 0 // scp_
scp main.cpp bgi902@172.16.64.19: / mnt / share1 / liusheng_test / work_v1 / hdcancer / bqsr /
#endif

//#include "r_d.hpp"
namespace r_d
{
	class td_br_arg
	{
		public:
			static vector<string>* p_vec_line;
			static vector<sam_record>* p_vec_line_sam_record;
			static string batch_rname;
			int start;
			int end;
			std::thread tid;

			grp_tables *p_e_grp_table;
			unordered_map<string, vector<bed_record>>* p_map_bed_chr_interval;
			vector<globalB::vcf_range*>* p_vec_vcf_overlap_p_range;
			string *p_hg19_path;
			string *p_fn_vcf;

			//td_br_arg (vector<string> &vec_line_, vector<sam_record> &vec_line_sam_record_, int &start_, int &end_, grp_tables& e_grp_table_, unordered_map<string, vector<bed_record>>& map_bed_chr_interval_, string& hg19_path_, vector<string>& vec_fn_vcf_);
			td_br_arg(vector<string> &vec_line_, vector<sam_record> &vec_line_sam_record_, int &start_, int &end_, grp_tables& e_grp_table_, unordered_map<string, vector<bed_record>>& map_bed_chr_interval_, string& hg19_path_, string& fn_vcf_, vector<globalB::vcf_range*>& vec_vcf_overlap_range_);

			td_br_arg();
	};

	class td_pr_arg
	{
		public:
			static vector<string>* p_vec_line;
			static string batch_rname;
			static grp_tables *p_e_grp_table;

			std::thread tid;
			int start;
			int end;
			string e_batch_str_sam;
			td_pr_arg(vector<string>& vec_line_, int& start_, int& end_, grp_tables& e_grp_table_);
			td_pr_arg();
	};


#if !AUTOTEST
	const int e_batch_num_br = E_BATCH_NUM_BR; //read number
	const int td_num_br = TD_NUM_BR; //thread number
	const int e_batch_num_pr = E_BATCH_NUM_PR;
	const int td_num_pr = TD_NUM_PR;
#endif

#if AUTOTEST
	const int e_batch_num_br = atoi(getenv("E_BATCH_NUM_BR"));
	const int td_num_br = atoi(getenv("TD_NUM_BR"));
	const int e_batch_num_pr = atoi(getenv("E_BATCH_NUM_PR"));
	const int td_num_pr = atoi(getenv("TD_NUM_PR"));
#endif

	const int batch_size_br = td_num_br * e_batch_num_br;
	const int batch_size_pr = td_num_pr * e_batch_num_pr;
	const int MAX_E_LINE = 1024;
	const int e_line_sz_value = 250;
	const int chr_field_no = 3;


	vector<string> vec_line{};
	vector<sam_record> vec_line_sam_record{};
	vector<vector<globalB::vcf_range*>> arr_vec_overlap_p_range{};

	vector<r_d::td_br_arg> vec_td_br_arg{};
	vector<r_d::td_pr_arg> vec_td_pr_arg{};
	//vector<string> vec_batch_rname{};

	int if_pre_next_eline_chr_not_equ(string &chr_old, string &e_line_new, string& chr_new);
	void td_br_exec(r_d::td_br_arg &e_td_arg);
	void td_br_exec_test(r_d::td_br_arg &e_td_arg);
	void td_pr_exec(r_d::td_pr_arg &e_td_arg);
	void read_sam_dispatch_2_td_for_br(const int e_batch_num, const int td_num, string &fn_sam, unordered_map<string, vector<bed_record>>& map_bed_chr_interval, string& hg19_path, string& fn_vcf);
	void read_sam_dispatch_2_td_for_pr(const int e_batch_num, const int td_num, string &fn_sam, string& fn_out_sam, grp_tables& e_grp_table);
	string& set_batch_chr_from_vec_line(vector<string>& vec_line);

};

//#include "r_d.cpp"
// r_d__ start
vector<string>* r_d::td_br_arg::p_vec_line = NULL;
vector<sam_record>* r_d::td_br_arg::p_vec_line_sam_record = NULL;
string r_d::td_br_arg::batch_rname = "INITIAL_CHRNAME";
r_d::td_br_arg::td_br_arg()
{
	// empty do
}

r_d::td_br_arg::td_br_arg(vector<string> &vec_line_, vector<sam_record> &vec_line_sam_record_, int &start_, int &end_, grp_tables& e_grp_table_, unordered_map<string, vector<bed_record>>& map_bed_chr_interval_, string& hg19_path_, string& fn_vcf_, vector<globalB::vcf_range*>& vec_vcf_overlap_p_range_)
{


	start = start_;
	end = end_;

	p_hg19_path = NULL;
	p_fn_vcf = NULL;
	p_e_grp_table = NULL;
	p_map_bed_chr_interval = NULL;
	p_vec_vcf_overlap_p_range = NULL;

	assert(NULL == p_map_bed_chr_interval);

	p_e_grp_table = &e_grp_table_;
	p_map_bed_chr_interval = &map_bed_chr_interval_;
	p_hg19_path = &hg19_path_;
	p_fn_vcf = &fn_vcf_;
	p_vec_vcf_overlap_p_range = &vec_vcf_overlap_p_range_;

	//------------------
	static int static_first_td_data_arg = 0;
	if (static_first_td_data_arg++ == 0)
	{
		r_d::td_br_arg::p_vec_line = NULL;
		r_d::td_br_arg::p_vec_line_sam_record = NULL;
		assert(NULL == r_d::td_br_arg::p_vec_line);
		r_d::td_br_arg::p_vec_line = &vec_line_;
		r_d::td_br_arg::p_vec_line_sam_record = &vec_line_sam_record_;
	}

}

// r_d pr arg
vector<string>* r_d::td_pr_arg::p_vec_line = NULL;
grp_tables* r_d::td_pr_arg::p_e_grp_table = NULL;
r_d::td_pr_arg::td_pr_arg()
{
}

r_d::td_pr_arg::td_pr_arg(vector<string>& vec_line_, int& start_, int& end_, grp_tables& e_grp_table_)
{
	static int static_first_td_pr_arg = 0;
	if (static_first_td_pr_arg++ == 0)
	{

		r_d::td_pr_arg::p_e_grp_table = NULL;
		r_d::td_pr_arg::p_vec_line = NULL;

		assert(NULL == r_d::td_pr_arg::p_vec_line);

		r_d::td_pr_arg::p_e_grp_table = &e_grp_table_;
		r_d::td_pr_arg::p_vec_line = &vec_line_;
	}
	start = start_;
	end = end_;
}

int r_d::if_pre_next_eline_chr_not_equ(string &chr_old, string &e_line_new, string& chr_new)
{
	if (e_line_new.size() <= 1)
	{
		return 1;
	}

	const char delimiter = '\t';
	std::stringstream  data_new(e_line_new);

	for (int i = 0; i < r_d::chr_field_no; i++)
	{
		std::getline(data_new, chr_new, delimiter);
	}

	int flag = chr_old == chr_new ? 0 : 1;
	chr_old = chr_new;
	return flag;
}

void r_d::td_br_exec(r_d::td_br_arg &e_td_arg)
{
	//cout << r_d::td_br_arg::batch_rname << endl;
	auto &_vec_line = *(e_td_arg.p_vec_line);
	auto &_vec_line_sam_record = *(e_td_arg.p_vec_line_sam_record);
	auto &map_bed_chr_interval = *(e_td_arg.p_map_bed_chr_interval);
	auto &hg19_path = *(e_td_arg.p_hg19_path);
	auto &fn_vcf = *(e_td_arg.p_fn_vcf);
	auto &e_grp_table = *(e_td_arg.p_e_grp_table);
	auto &e_vec_vcf_overlap_p_range = *(e_td_arg.p_vec_vcf_overlap_p_range);

	for (int i = e_td_arg.start; i < e_td_arg.end; i++)
	{
		auto &_e_line = _vec_line[i];
		_vec_line_sam_record[i] = sam_record(_e_line);
	}

#if 1
	static int chr_cnt = 0;
	static int read_cnt = 0;
	volatile clock_t brBeginT, filterPreprocessT, aggregateReadDataT, forloopT,
		 forloopBeginT, baqT, knownSitesT, isSNPT, skipT, snpErrorsT,
		 computeCovariatesT, updateDataForReadT;
	chr_cnt++;
	static vector<float> chr_time(4, 0);
	static vector<float> read_time(6, 0);
	/**
	 * first method: take map_vec_sam_record_Raw[rname] as an argument, when no lock, consistency is ok!.(test for sure)
	 * second method: directly take map_vec_sam_record_Raw as an argument, when no lock and we update value in sam_record, consistency is not ok!.(test for sure)
	 * so is it right we use the first method?
	 * I have already fixed problem in second method by cache cigar lately in aggregate_ReadData.
	 */
	brBeginT = clock();

	vector<sam_record*> filtered_vec_p_sam_record = grptools::filterPreprocess_v1(e_td_arg.start, e_td_arg.end, _vec_line_sam_record, map_bed_chr_interval[r_d::td_br_arg::batch_rname]);
	filterPreprocessT = clock();

	if (filtered_vec_p_sam_record.size() == 0) return;

	vector<sam_record*> vec_p_sam_record = grptools::aggregateReadData(filtered_vec_p_sam_record, r_d::td_br_arg::batch_rname, e_vec_vcf_overlap_p_range, hg19_path, fn_vcf);

	aggregateReadDataT = clock();

	int vec_sam_record_sz = (int)vec_p_sam_record.size();
	assert(vec_sam_record_sz != 0);

	// { // start br loop //bthr_
	int nth = 0;
	grptools::grpTempDat * grpTempDatArr = (grptools::grpTempDat *)malloc(sizeof(grptools::grpTempDat) * vec_sam_record_sz);
	assert(NULL != grpTempDatArr);
	for (auto & p_e_sam_record : vec_p_sam_record)
	{
		auto &e_sam_record = *p_e_sam_record;
		forloopBeginT = clock();
		read_cnt++;
		string baq = baqtools::getBAQTag(e_sam_record, e_sam_record.refSeq);
		baqT = clock();

		vector<bool> knownSites = grptools::calculateKnownSitesByFeatures(e_sam_record, e_sam_record.bindings);
		knownSitesT = clock();
		vector<int> isSNP = grptools::calculateIsSNP(e_sam_record, e_sam_record.refSeq);
		isSNPT = clock();

		if (baq.length() != 0)
		{
			grptools::calculateSkipArray_v1(e_sam_record, knownSites, grpTempDatArr, nth);
			skipT = clock();
			grptools::calculateFractionalErrorArray_v1(isSNP, baq, grpTempDatArr, nth);
			snpErrorsT = clock();
			grptools::ComputeCovariates_v2(e_sam_record, grpTempDatArr, nth);
			grpTempDatArr[nth].length = (int)e_sam_record.seq.size();
			computeCovariatesT = clock();
			nth++;
		}


	}
	forloopT = clock();

	e_grp_table.updateDataForRead_v3(grpTempDatArr, vec_sam_record_sz);
	updateDataForReadT = clock();

#if !FLAG_USE_MULTI_THREAD
	//cout << "---------" << r_d::td_br_arg::batch_rname << " time ------------" << endl;
	chr_time[0] += filterPreprocessT - brBeginT;
	chr_time[1] += aggregateReadDataT - filterPreprocessT;
	chr_time[2] += forloopT - aggregateReadDataT;
	chr_time[3] += updateDataForReadT - forloopT;
	cout << "id:" << chr_cnt << ","
		<< "total reads:" << read_cnt << ","
		<< "filterPreprocessT:" << chr_time[0] / (CLOCKS_PER_SEC * read_cnt) << ","
		<< "aggregateReadDataT:" << chr_time[1] / (CLOCKS_PER_SEC * read_cnt) << ","
		<< "forloopT:" << chr_time[2] / (CLOCKS_PER_SEC * read_cnt) << ","
		<< "updateDataForReadT:" << chr_time[3] / (CLOCKS_PER_SEC * read_cnt)
		<< endl;
	//cout << "---------" << r_d::td_br_arg::batch_rname << " time ------------" << endl;
#endif
	// }   // end br loop

	//cout << endl << "----------------------" << endl << endl;
#endif
}

void r_d::td_pr_exec(r_d::td_pr_arg &e_td_arg)
{
	auto &_vec_line = *(e_td_arg.p_vec_line);
	auto &e_grp_table = *(e_td_arg.p_e_grp_table);

	string &e_batch_str_sam = e_td_arg.e_batch_str_sam;
	e_batch_str_sam.clear();

	volatile clock_t printReadsBeginT, printReadsComputeCovariatesT, performSequentialQualityCalculationT;
	vector<float> printReadsTime50(2, 0);
	static int s_cnt_r = 0;

	static vector<int> keyVec;
	int recalibratedQualityScore, qualFromRead, contextIdex, cycleIdex;
	double deltaQReported, deltaQCovariates, recalibratedQual;


	for (int j = e_td_arg.start; j < e_td_arg.end; j++)
	{
		auto &_e_line = _vec_line[j];

		sam_record e_sam_record(_e_line);
		//continue;

		// {	// start pthr_ loop

		if (bgitools::MalformedReadFilter(e_sam_record))
		{
			continue;
		}

		printReadsBeginT = clock();
		s_cnt_r++;

		vector<struct grptools::CVKEY> readCovariates = grptools::ComputeCovariates_v1(e_sam_record);

		printReadsComputeCovariatesT = clock();
		vector<bgitools::EventType> etVec{ bgitools::EventType::BASE_SUBSTITUTION,
			bgitools::EventType::BASE_INSERTION,
			bgitools::EventType::BASE_DELETION };

		int readLength = (e_sam_record).seq.length();
		for (bgitools::EventType eventModel : etVec)
		{
			switch (eventModel)
			{
				case bgitools::EventType::BASE_SUBSTITUTION:
					{
						for (int offset = 0; offset < readLength; offset++)
						{
							grptools::CVKEY keySet = readCovariates[offset]; // get the keyset for this base using the error model
							qualFromRead = keySet.quality;
							if (qualFromRead >= globalB::MIN_USABLE_Q_SCORE) {
#if dq_adv
								contextIdex = globalB::context_v_k_Map[keySet.contextKey];
								cycleIdex = globalB::cycle_v_k_Map[keySet.cycleKey];
								deltaQReported = globalB::map_deltaQReported[qualFromRead];
								deltaQCovariates = globalB::deltaQCovariatesArr[qualFromRead - globalB::MIN_USABLE_Q_SCORE][contextIdex][cycleIdex];
								recalibratedQual = qualFromRead + globalB::globalDeltaQ + deltaQReported + deltaQCovariates;
								int rq = round(recalibratedQual);
								recalibratedQualityScore = max(min(rq, globalB::MAX_PHRED_SCORE), 1);
#else
								keyVec = { qualFromRead, keySet.contextKey, keySet.cycleKey };
								recalibratedQualityScore = bqsrtools::PerformSequentialQualityCalculation(
										e_grp_table,
										keyVec,
										eventModel); // recalibrate the base
#endif
								qualFromRead = recalibratedQualityScore;
							}
							(e_sam_record).qual[offset] = (qualFromRead + 33);
						}
						break;
					}

				default: {
						 cout << "unknown error!" << endl;
						 assert(0 == 1);
					 }
			}

			break;
		}
		performSequentialQualityCalculationT = clock();

#if !FLAG_USE_MULTI_THREAD
		printReadsTime50[0] += printReadsComputeCovariatesT - printReadsBeginT;
		printReadsTime50[1] += performSequentialQualityCalculationT - printReadsComputeCovariatesT;
		if (s_cnt_r % 100 == 0)
		{
			cout << "id:" << s_cnt_r << ","
				<< "printReadsComputeCovariatesT:" << printReadsTime50[0] / (CLOCKS_PER_SEC * s_cnt_r) << ","
				<< "performSequentialQualityCalculationT:" << printReadsTime50[1] / (CLOCKS_PER_SEC * s_cnt_r)
				<< endl;
		}
#endif

		e_batch_str_sam += e_sam_record.to_string();
		// } // end pr loop
	}

}

void r_d::read_sam_dispatch_2_td_for_br(const int e_batch_num, const int td_num, string &fn_sam, unordered_map<string, vector<bed_record>>& map_bed_chr_interval, string& hg19_path, string& fn_vcf)
{
	volatile clock_t br_dispatch_beginT, br_dispatch_readT, br_multiProcessT;
	vector<float> br_dispatch_time(3, 0);

	uint64_t max_chr_bytes = 0;
	globalB::init_vcf_idx_2_map(fn_vcf, max_chr_bytes);
	globalB::alloc_bytes_of_arr_vcf_range(max_chr_bytes);

	auto batch_size = e_batch_num * td_num;
	r_d::vec_line.resize(batch_size);
	r_d::vec_line_sam_record.resize(batch_size);
	r_d::vec_td_br_arg.resize(td_num);

	r_d::arr_vec_overlap_p_range.resize(td_num);
	const int max_overlap_range = 1 * 1024 * 1024;  // experiment value 

	string e_mid_chrR_chrS;
	map<string, int> map_batch_rname{};

	// for critical area, sometime, it end without count to batch_size
	int true_read_num = batch_size;
	int already_read_num = 0;

	int cnt_batch_num = 0;
	int cnt_line_num = 0;
	size_t flag_first_time = 0;
	int flag_meet_new_chr = 0;

	ifstream if_(fn_sam);  assert(if_.is_open());

	string chr_old = "initial_chrname";
	string chr_new;

	string e_line;
	if_.seekg(0, ios::end);

	size_t fn_sam_sz = (size_t)if_.tellg();
	int file_need_batch_num = (int)(fn_sam_sz / r_d::e_line_sz_value / batch_size + 1);

	// read header of sam
	sam_record::header_str = string("");
	sam_record::group_name = string("");
	sam_record::BI = string("");
	sam_record::BD = string("");

	if_.seekg(0, ios::beg);
	while (!if_.eof())
	{
		std::getline(if_, e_line);

		if (e_line[0] == '@')
		{
			sam_record::header_str += e_line + "\n";
			continue;
		}


		if (e_line[0] == 'C')
		{

#if __linux__
			if_.seekg(if_.tellg() - (streampos)e_line.size() - 1);
#else
			if_.seekg(if_.tellg() - (streampos)e_line.size() - 2);
#endif

			auto first_e_sam_record = sam_record(e_line);
			chr_old = first_e_sam_record.rname;

			// from header to get  group_name
			auto *target_sm = "\tSM:";
			auto len_target = strlen(target_sm);
			auto loc_sm = sam_record::header_str.find(target_sm);
			assert(loc_sm != string::npos);
			auto loc_end = sam_record::header_str.find('@', loc_sm);
			sam_record::group_name = sam_record::header_str.substr(loc_sm + len_target, loc_end - loc_sm - len_target - 1);

			// get BD BI, to static var
			auto len_qual = first_e_sam_record.qual.size();
			sam_record::BI = "BI:Z:" + string(len_qual, 'N');
			sam_record::BD = "BD:Z:" + string(len_qual, 'N');

			break;
		}
	}

	if (flag_first_time++ == 0)
	{

		for (int i = 0; i < batch_size; i++)
		{
			r_d::vec_line[i].resize(r_d::MAX_E_LINE);
		}

		r_d::arr_vec_overlap_p_range[0].resize(40 * max_overlap_range);
		for (int i = 1; i < td_num; i++)
		{
			r_d::arr_vec_overlap_p_range[i].resize(max_overlap_range);
		}
	}

	while (!if_.eof())
	{
		br_dispatch_beginT = clock();
		true_read_num = batch_size;
		cnt_line_num = 0;
		int true_td_num = td_num;
		int left_read_num = 0;
		if (1 == flag_meet_new_chr)
		{
			r_d::vec_line[0] = e_mid_chrR_chrS;
			cnt_line_num++;
		}

		flag_meet_new_chr = 0;
		while (cnt_line_num < batch_size && !if_.eof())
		{
			std::getline(if_, r_d::vec_line[cnt_line_num]);

			if (cnt_line_num > 0)
			{
				auto &vec_line_new = r_d::vec_line[cnt_line_num];
				if (flag_meet_new_chr = if_pre_next_eline_chr_not_equ(chr_old, vec_line_new, chr_new))
				{
					e_mid_chrR_chrS = vec_line_new;
					true_read_num = cnt_line_num;
					true_td_num = (int)(true_read_num / e_batch_num);
					left_read_num = true_read_num % e_batch_num;
					break;
				}
			}

			cnt_line_num++;
		}

		// deal with vector data
		cnt_batch_num++;
		assert(true_td_num * e_batch_num + left_read_num == true_read_num);
		already_read_num += true_read_num;
		//        cout << "- br already_read_num : " << already_read_num << endl;



		int flag_run_data = 0;
#if 0
		const int run_from_num = 78535213;
		if (already_read_num >= run_from_num)
		{
			flag_run_data = 1;
		}
#endif
#if 1
		flag_run_data = 1;
#endif
		if (flag_run_data)
		{
			{ //set up args

				for (int k = 0; k < true_td_num; k++)
				{
					auto start_ = k * e_batch_num;
					auto end_ = start_ + e_batch_num;
					r_d::vec_td_br_arg[k] = r_d::td_br_arg(r_d::vec_line, r_d::vec_line_sam_record, start_, end_, globalB::grp_table_arr[k], map_bed_chr_interval, hg19_path, fn_vcf, r_d::arr_vec_overlap_p_range[k]);
				}

				if (left_read_num)
				{
					auto start_ = true_td_num * e_batch_num;
					auto end_ = true_td_num * e_batch_num + left_read_num;
					r_d::vec_td_br_arg[true_td_num] = r_d::td_br_arg(r_d::vec_line, r_d::vec_line_sam_record, start_, end_, globalB::grp_table_arr[true_td_num], map_bed_chr_interval, hg19_path, fn_vcf, r_d::arr_vec_overlap_p_range[true_td_num]);
					true_td_num++;	// td num should add left block
				}

				auto &batch_chrname = r_d::set_batch_chr_from_vec_line(r_d::vec_line);
				auto len_chr_old = map_batch_rname.size();
				map_batch_rname[batch_chrname] = 1;
				auto len_chr_new = map_batch_rname.size();

				if (len_chr_old != len_chr_new)
				{
					faidx_t *hg19fai = fai_load(hg19_path.c_str());
					if (!hg19fai)
					{
						fprintf(stderr, "Could not load fai index of %s\n", hg19_path.c_str());
						assert(0 == 1);
					}
					globalB::load_chromosome(r_d::td_br_arg::batch_rname, hg19fai);
					fai_destroy(hg19fai);

					globalB::load_vcf_range_by_chrname(r_d::td_br_arg::batch_rname); //if not found, return 0
					globalB::vcf_record_uniq();
				}

			}

			br_dispatch_readT = clock();
			if (!FLAG_USE_MULTI_THREAD)
			{
				for (int i = 0; i < true_td_num; i++)
				{
					auto &e_td_arg = r_d::vec_td_br_arg[i];
					e_td_arg.tid = std::thread(r_d::td_br_exec, std::ref(e_td_arg));
					e_td_arg.tid.join();
				}
			}


			if (FLAG_USE_MULTI_THREAD)
			{
				for (int i = 0; i < true_td_num; i++)
				{
					auto &e_td_arg = r_d::vec_td_br_arg[i];
					e_td_arg.tid = std::thread(r_d::td_br_exec, std::ref(e_td_arg));
				}

				for (int i = 0; i < true_td_num; i++)
				{
					auto &e_td_arg = r_d::vec_td_br_arg[i];
					e_td_arg.tid.join();
				}
			}
		}

		br_multiProcessT = clock();
		//cout << "---------br dispatch time start------------" << endl;
		br_dispatch_time[0] += br_dispatch_readT - br_dispatch_beginT;
		br_dispatch_time[1] += br_multiProcessT - br_dispatch_readT;
		br_dispatch_time[2] += br_multiProcessT - br_dispatch_beginT;
#if 0
		cout << "br_dispatch_readT:" << br_dispatch_time[0] / (CLOCKS_PER_SEC * already_read_num) << ","
			<< "br_multiProcessT:" << br_dispatch_time[1] / (CLOCKS_PER_SEC * already_read_num) << ","
			<< "totalT:" << br_dispatch_time[2] / (CLOCKS_PER_SEC * already_read_num) << ","
			<< endl;
#endif 
		//cout << "---------br dispatch time end------------" << endl;
	}
	if_.close();

	r_d::vec_line_sam_record.resize(0);
	r_d::vec_td_br_arg.resize(0);


	for (int i = 0; i< td_num; i++)
	{
		r_d::arr_vec_overlap_p_range[i].clear();
	}
	r_d::arr_vec_overlap_p_range.clear();

	globalB::delete_bytes_of_arr_vcf_range();
}

void r_d::read_sam_dispatch_2_td_for_pr(const int e_batch_num, const int td_num, string& fn_sam, string& fn_out_sam, grp_tables& e_grp_table)
{
	//    volatile clock_t printReadsBeginT, printReadsComputeCovariatesT, performSequentialQualityCalculationT;
	//    vector<float> printReadsTime50(2, 0);
	//    static int s_cnt_r = 0;

	auto batch_size = e_batch_num * td_num;
	r_d::vec_line.resize(batch_size);
	r_d::vec_td_pr_arg.resize(td_num);

	string e_mid_chrR_chrS;

	// for critical area, sometime, it end without count to batch_size
	int true_read_num = batch_size;
	int already_read_num = 0;

	int cnt_batch_num = 0;
	int cnt_line_num = 0;
	size_t flag_first_time = 0;
	int flag_meet_new_chr = 0;

	ofstream of_(fn_out_sam, ios::app); assert(of_.is_open());

	ifstream if_(fn_sam);  assert(if_.is_open());

	string chr_old = "initial_chrname";
	string chr_new;

	string e_line;
	if_.seekg(0, ios::end);

	size_t fn_sam_sz = (size_t)if_.tellg();
	int file_need_batch_num = (int)(fn_sam_sz / r_d::e_line_sz_value / batch_size + 1);

	if (sam_record::header_str == "initial_sam_header_str")
	{
		sam_record::header_str = string("");
		sam_record::group_name = string("");
		sam_record::BI = string("");
		sam_record::BD = string("");
	}

	if_.seekg(0, ios::beg);
	// skip header of sam
	while (!if_.eof())
	{
		std::getline(if_, e_line);

		if (e_line[0] == '@')
		{
			if (DEBUG && !br_Ana && pr_Ana)
			{
				sam_record::header_str += e_line + "\n";
			}
			continue;
		}

		of_ << sam_record::header_str;

		if (e_line[0] == 'C')
		{

#if __linux__
			if_.seekg(if_.tellg() - (streampos)e_line.size() - 1);
#else
			if_.seekg(if_.tellg() - (streampos)e_line.size() - 2);
#endif
			auto first_e_sam_record = sam_record(e_line);
			chr_old = first_e_sam_record.rname;

			if (sam_record::BI == "")
			{
				// from header to get  group_name
				auto *target_sm = "\tSM:";
				auto len_target = strlen(target_sm);
				auto loc_sm = sam_record::header_str.find(target_sm);
				assert(loc_sm != string::npos);
				auto loc_end = sam_record::header_str.find('@', loc_sm);
				sam_record::group_name = sam_record::header_str.substr(loc_sm + len_target, loc_end - loc_sm - len_target - 1);

				// get BD BI, to static var
				auto len_qual = first_e_sam_record.qual.size();
				sam_record::BI = "BI:Z:" + string(len_qual, 'N');
				sam_record::BD = "BD:Z:" + string(len_qual, 'N');
			}

			break;
		}
	}

	if (flag_first_time++ == 0)
	{
		for (int i = 0; i < batch_size; i++)
		{
			r_d::vec_line[i].resize(r_d::MAX_E_LINE);
		}
	}

	while (!if_.eof())
	{
		true_read_num = batch_size;
		cnt_line_num = 0;
		int true_td_num = td_num;
		int left_read_num = 0;
		if (1 == flag_meet_new_chr)
		{
			r_d::vec_line[0] = e_mid_chrR_chrS;
			cnt_line_num++;
		}

		flag_meet_new_chr = 0;
		while (cnt_line_num < batch_size && !if_.eof())
		{
			std::getline(if_, r_d::vec_line[cnt_line_num]);

			if (cnt_line_num > 0)
			{
				auto &vec_line_new = r_d::vec_line[cnt_line_num];

				if (flag_meet_new_chr = if_pre_next_eline_chr_not_equ(chr_old, vec_line_new, chr_new))
				{
					e_mid_chrR_chrS = vec_line_new;
					true_read_num = cnt_line_num;
					true_td_num = (int)(true_read_num / e_batch_num);
					left_read_num = true_read_num % e_batch_num;
					break;
				}
			}

			cnt_line_num++;
		}

		// deal with vector data
		cnt_batch_num++;
		assert(true_td_num * e_batch_num + left_read_num == true_read_num);
		already_read_num += true_read_num;
		//		cout << "- pr already_read_num : " << already_read_num << endl;


		int flag_run_data = 0;
#if 0
		const int run_from_num = 78535213;
		if (already_read_num >= run_from_num)
		{
			flag_run_data = 0;
		}
#endif
#if 1
		flag_run_data = 1;
#endif
		if (flag_run_data)
		{
			{ //set up args
				for (int k = 0; k < true_td_num; k++)
				{
					auto start_ = k * e_batch_num;
					auto end_ = start_ + e_batch_num;
					r_d::vec_td_pr_arg[k] = r_d::td_pr_arg(r_d::vec_line, start_, end_, e_grp_table);
				}

				if (left_read_num)
				{
					auto start_ = true_td_num * e_batch_num;
					auto end_ = true_td_num * e_batch_num + left_read_num;

					r_d::vec_td_pr_arg[true_td_num] = r_d::td_pr_arg(r_d::vec_line, start_, end_, e_grp_table);
					true_td_num++;	// td num should add left block
				}

				auto &batch_chrname = r_d::set_batch_chr_from_vec_line(r_d::vec_line);
			}

			if (!FLAG_USE_MULTI_THREAD)
			{
				for (int i = 0; i < true_td_num; i++)
				{
					auto &e_td_arg = r_d::vec_td_pr_arg[i];
					e_td_arg.tid = std::thread(r_d::td_pr_exec, std::ref(e_td_arg));
					e_td_arg.tid.join();

				}
			}


			if (FLAG_USE_MULTI_THREAD)
			{
				for (int i = 0; i < true_td_num; i++)
				{
					auto &e_td_arg = r_d::vec_td_pr_arg[i];
					e_td_arg.tid = std::thread(r_d::td_pr_exec, std::ref(e_td_arg));
				}

				for (int i = 0; i < true_td_num; i++)
				{
					auto &e_td_arg = r_d::vec_td_pr_arg[i];
					e_td_arg.tid.join();
				}
			}
			// after run
			for (int i = 0; i<true_td_num; i++)
			{
				auto &e_td_arg = r_d::vec_td_pr_arg[i];
				of_ << e_td_arg.e_batch_str_sam;
			}
		}
	}
	if_.close();

	of_.close();
}

string& r_d::set_batch_chr_from_vec_line(vector<string>& vec_line)
{
	time_t start, br_time, pr_time, end_time;

	auto &e_line = vec_line[0];
	assert(e_line.size()>1);

	const char delimiter = '\t';
	std::stringstream  data(e_line);

	auto &rname = r_d::td_br_arg::batch_rname;
	for (int i = 0; i < r_d::chr_field_no; i++)
	{
		std::getline(data, rname, delimiter);
	}

	assert(rname == "*" || (rname[0] == 'c' && rname[1] == 'h' && rname[2] == 'r'));
	return rname;
}

// r_d__ end

static inline int check_sam_write1_bqsr(samFile *fp, const bam_hdr_t *h, const bam1_t *b, const char *fname, int *retp)
{
	int r = sam_write1(fp, h, b);
	if (r >= 0) return r;

	if (fname) { printf("view", "writing to \"%s\" failed", fname); assert(0 == 1); }
	else { printf("view", "writing to standard output failed"); assert(0 == 1); }

	*retp = EXIT_FAILURE;
	return r;
}



//#include "main.cpp"

#endif 

#if 0
typedef struct
{   
	static int s_int; 
	static const int XXX = 99; 
	int x;
	int y;
} xxx;

int xxx::s_int = 9999; 

class td_args
{
	public:

		int i; 
		int *p;

		td_args() 
		{
			i = 0;
			p = NULL;
		};
		td_args(int i_)
		{
			i = i_;
			p = new int[i];
		}
		~td_args()
		{

			DEL_ARR_MEM(p);
			i = 0;
		}

};



void td_exec_0(td_args &td_arg_)
{  

	for (int i = 0; i < 1e7; i++)
	{
		int t = i * i;
	}
	cout << td_arg_.i << endl;


	int x = 0; 
	int y = 1; 

	auto td_0_0 = std::thread([](int &x, int &y) {printf("- 0: %d, %d\n", x, y); }, std::ref(x), std::ref(y));
	auto td_0_1 = std::thread([](int &x, int &y) {printf("- 1: %d, %d\n", x, y); }, std::ref(x), std::ref(y));

	td_0_0.join();
	td_0_1.join();




}

void td_exec_1(td_args &td_arg_)
{

	for (int i = 0; i < 1e6; i++)
	{
		int t = i * i;
	}


	cout << td_arg_.i << endl; 
}

/* Converts a hex character to its integer value */
char from_hex(char ch) {
	return isdigit(ch) ? ch - '0' : tolower(ch) - 'a' + 10;
}

/* Converts an integer value to its hex character*/
char to_hex(char code) {
	static char hex[] = "0123456789ABCDEF";
		return hex[code & 15];
}

/* Returns a url-encoded version of str */
/* IMPORTANT: be sure to free() the returned string after use */
unsigned char *url_encode(unsigned char *str, unsigned int & len) {
	unsigned char *pstr = str, *buf = (unsigned char*)malloc(len * 3 + 1), *pbuf = buf;
		for (int i = 0; i < len; i++)
		{
			
				if (isalnum(*pstr) || *pstr == '-' || *pstr == '_' || *pstr == '.' || *pstr == '~')
					*pbuf++ = *pstr;
				else if (*pstr == ' ')
					*pbuf++ = '+';
				else
					*pbuf++ = '%', *pbuf++ = to_hex(*pstr >> 4), *pbuf++ = to_hex(*pstr & 15);
						pstr++;    
		}
	
		*pbuf = '\0';
		cout << len * 3 + 1 << endl; 
		cout << pbuf - buf << endl;
		//assert(len * 3 + 1 == pbuf - buf);
		len = pbuf - buf;
		
		return buf;
}

/* Returns a url-decoded version of str */
/* IMPORTANT: be sure to free() the returned string after use */
char *url_decode(char *str, int len) {
	char *pstr = str, *buf = (char*)malloc(strlen(str) + 1), *pbuf = buf;
		while (*pstr) {
			if (*pstr == '%') {
				if (pstr[1] && pstr[2]) {
					*pbuf++ = from_hex(pstr[1]) << 4 | from_hex(pstr[2]);
						pstr += 2;
				}
			}
			else if (*pstr == '+') {
				*pbuf++ = ' ';
			}
			else {
				*pbuf++ = *pstr;
			}
			pstr++;
		}
	*pbuf = '\0';
		return buf;
}


#endif 

template<typename T> string tools_to_string(const T &t)
{
    ostringstream ss;
    ss << t;
    return ss.str();
}

#if 0
void exec_e(float &e)
{
    cout << e << " ";
}



void exec_multi_threads_on_data(vector<float>& arr_src, const int& N, const int& td_num, void (*cb)(float &e))
{
#if 1
    //typedef  pair<int, std::thread> i_td;
    int td_num_real = td_num - 1; 
    int mod_num = (int)(N / td_num);
    vector< pair<int, std::thread> > arr_pair_i_td{};

    //const int N = (const int)(2e2 + 3);
    //int mod_num = 3;
    arr_pair_i_td.resize(N % mod_num == 0 ? N / mod_num : N / mod_num + 1);

    int cnt = 0;
    int cnt_arr = 0;

    cnt = 0;
    while (cnt < N)
    {

        //cout << cnt << endl;

        int id_td = (int)(cnt / mod_num);
        cout << id_td << endl;
        auto& pair_i_i = make_pair(cnt, cnt + mod_num > N ? N : cnt + mod_num);


        arr_pair_i_td[id_td].first = id_td;
        arr_pair_i_td[id_td].second = thread([cb](int id_td, vector<float>& arr_src, pair<int, int> pair_i_i) {

#if 1
            float sleep_time_seconds = id_td * 0.1f;
            string cmd_sleep = string("sleep ") + tools_to_string(sleep_time_seconds) + "s";
            system(cmd_sleep.c_str());
            //cout << cmd_sleep << endl; 
#endif 
            cout << "- id_td: " << id_td << endl;
            
            for (auto i = pair_i_i.first; i < pair_i_i.second; i++)
            {
                //cout << arr_src[i] << " ";
                cb(arr_src[i]);
            }
            cout << endl << "--------------------" << endl;

        },
            id_td, std::ref(arr_src), pair_i_i
            );



        //cnt_arr++;
        cnt += mod_num;


        //arr_pair_i_td[id_td].second.join();
    }


    for (auto& e_pair : arr_pair_i_td)
    {
        //cout << e_pair.first << endl; 
        e_pair.second.join();
    }


    arr_pair_i_td.clear();


#endif

}


struct A
{
 

}; 
#endif 

#if 0
vector<string> split_str_2_vec(const string & str, const char & delimiter = '\t', const int & retainNum = -1, int flag_push_tail_str = 0);
vector<string> split_str_2_vec(const string & str, const char & delimiter, const int & retainNum, int flag_push_tail_str)
{
    std::vector<std::string>   vec_ret{};
    vec_ret.reserve(16);

    std::stringstream  data(str);

    std::string line;
    if (retainNum == -1)
    {
        if (delimiter == ' ')
        {
            while (std::getline(data, line, delimiter)) // assume
            {
                // Note: if multiple delimitor in the source string,
                //           you may see many empty item in vector
                if (line == "") continue;
                vec_ret.push_back(line);
            }
            return vec_ret;
        }
        while (std::getline(data, line, delimiter))     // assume
        {
            // Note: if multiple delimitor in the source string,
            //       you may see many empty item in vector
            vec_ret.push_back(line);
        }
    }
    else
    {
        if (delimiter == ' ')
        {
            int cnt = 0;
            while (std::getline(data, line, delimiter)) // assume
            {
                if (cnt >= retainNum) break;
                if (line == "") continue;
                vec_ret.push_back(line);
                cnt++;
            }
            return vec_ret;
        }
        int cnt = 0;
        while (cnt < retainNum && std::getline(data, line, delimiter))     // assume
        {
            //if (cnt >= retainNum) break;
            vec_ret.push_back(line);
            cnt++;
        }
    }

    if (flag_push_tail_str)
    {
        std::getline(data, line, '\a');
        vec_ret.push_back(line);
    }
    return vec_ret;
}
#endif 


#if 0 
#define SIZE 1 * 1024 * 1024 * 1024

#define print_v_i(v)    ( for_each(v.begin(),v.end(),[](int i){cout<<i<<" ";} ) ) 



class B {
public:
    int x; 
    int y; 
    B() { x = 99; y = 999; cout << "- con" << endl; }
    ~B() { cout << "- decon" << endl; }
};

template<typename Func, typename... Args>
auto ret_f(Func func, Args... args)
{
    return [=](auto... lastParam)
    {
        return func(args..., lastParam...);
    };
};

#endif 

#if 0
int fac(int n, int sum)
{
    if (n == 0)
    {
        return sum;
    }
    return fac(n - 1, sum*n);
}

int fac(int n)
{
    return fac(n, 1);
}

/* permutation.cpp */
#include <iostream>

using namespace std;

// Calculation the permutation
// of the given string
void doPermute(
    const string &chosen,
    const string &remaining)
{
    if (remaining == "")
    {
        cout << chosen << endl;
    }
    else
    {
        for (uint32_t u = 0; u < remaining.length(); ++u)
        {
            doPermute(
                chosen + remaining[u],
                remaining.substr(0, u)
                + remaining.substr(u + 1));
        }
    }
}

// The caller of doPermute() function
void permute(const string &s)
{
    doPermute("", s);
}

vector<vector<char>> createLabyrinth()
{
    // Initializing the multidimensional vector
    // labyrinth 
    // # is a wall
    // S is the starting point
    // E is the finishing point
    vector<vector<char>> labyrinth =
    {
        { '#', '#', '#', '#', '#', '#', '#', '#' },
        { '#', 'S', ' ', ' ', ' ', ' ', ' ', '#' },
        { '#', '#', '#', ' ', '#', '#', '#', '#' },
        { '#', ' ', '#', ' ', '#', '#', '#', '#' },
        { '#', ' ', ' ', ' ', ' ', ' ', ' ', '#' },
        { '#', ' ', '#', '#', '#', '#', '#', '#' },
        { '#', ' ', ' ', ' ', ' ', ' ', 'F', '#' },
        { '#', '#', '#', '#', '#', '#', '#', '#' }
    };

    return labyrinth;
}

void displayLabyrinth(vector<vector<char>> labyrinth)
{
    cout << endl;
    cout << "====================" << endl;
    cout << "The Labyrinth" << endl;
    cout << "====================" << endl;
    auto rows = 8;
    auto cols = 8;

    // Displaying all characters in labyrinth vector
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cout << labyrinth[i][j] << " ";
        }
        cout << endl;
    }
    cout << "====================" << endl << endl;
}

#endif 
#if 0 

int *g_r_core = NULL;
auto range = [](int M, int N = 0)
{
    if (0 == N) { N = M; M = 0; }

    static size_t flag_first_range = 0;
    const int max_range = 1024 * 10;
    assert(N <= max_range);

    if (0 == flag_first_range++)
    {
        g_r_core = new int[max_range];  assert(NULL != g_r_core);
        for (int i = 0; i<max_range; i++) { g_r_core[i] = i; }
    }
    return vector<int>(g_r_core + M, g_r_core + N);
};

auto clear_range = []()
{
    if (NULL != g_r_core) { delete[] g_r_core; }
};



#endif 

#if 0
int filter_out_e(vector<string> & v_s)
{
    // 0, not filter out; 1, filter out
    int cnt_filter_out = 0;
    const int WINDOWSIZE = 64;
    const int WINDOWSTEP = 32;
    const float POINTFIVE = 0.5f;
    const float COMPLVAL = 7.0f;

    auto &seq = v_s[1];
    auto &qual = v_s[3];
    std::transform(seq.begin(), seq.end(), seq.begin(), ::toupper);
    auto &seqn = seq;
    auto length = seqn.size();
    auto steps = (int)((length - WINDOWSIZE) * 1.0f / WINDOWSTEP) + 1;
    auto rest = length - steps * WINDOWSTEP;

    auto num = WINDOWSIZE - 2;
    auto bynum = 1.0f / num;
    num--;
    auto mean = 0.0f;

    // cal "dust"
    float dustscore = 0.0f;
    vector<float> v_vals{};

    // 1
    for (auto &i : range(0, steps)) {
        auto str = seqn.substr((i * WINDOWSTEP), WINDOWSIZE);
        //%counts = ();
        unordered_map<string, int> counts;
        for (auto i : range(0, 62)) {
            counts[str.substr(i, 3)]++;
        }
        dustscore = 0;

        for (auto &e : counts)
        {
            auto &s_ = e.second;
            dustscore += (s_ * (s_ - 1) * POINTFIVE);
        }

        v_vals.push_back(dustscore * bynum);
    }

    // 2
    if (rest > 5)
    {
        auto str = seqn.substr((steps * WINDOWSTEP), rest);
        unordered_map<string, int> counts;
        num = rest - 2;

        for (auto i : range(0, num)) {
            counts[str.substr(i, 3)]++;
        }
        dustscore = 0;
        for (auto &e : counts)
        {
            auto &s_ = e.second;
            dustscore += (s_ * (s_ - 1) * POINTFIVE);
        }

        auto last_val = ((dustscore / (num - 1)) * ((WINDOWSIZE - 2) / (float)num));
        v_vals.push_back(last_val);
    }

    auto mean_ = accumulate(v_vals.begin(), v_vals.end(), 0.0) / v_vals.size();

    if ((int)(mean_ * 100 / 31) > COMPLVAL) 
    {
        cnt_filter_out = 1;
    }

    return cnt_filter_out;
};

#endif 

// st__ main_
int main(int argc, char **argv)
{

#if 0 
    const int e_line_sz = 4;
    string fn = "NULL";
    fn = "./data/18B0000399-1-11.umhg19.fq";
    if (argc > 1)
    {
        fn = string(argv[1]);
    }
    string fn_out_clean = fn + ".clean.fastq";
    ofstream of_(fn_out_clean.c_str()); assert(of_.is_open());
    ifstream if_(fn.c_str());  assert(if_.is_open());

    string e_str = "NULL";

    int cnt_lines = 0;
    int cnt_bad_base = 0;
    vector<string> v_str(e_line_sz, "");
    auto p_s = [&v_str,&of_]() {v_str[2] = string(v_str[0]); v_str[2][0] = '+'; for (auto &e : v_str) { of_ << e << endl; }};
    while (std::getline(if_, e_str))
    {
        //cout << "___" << e_str << "___" << endl; 
        auto v_idx = cnt_lines < e_line_sz ? cnt_lines : cnt_lines % 4;
        cnt_lines++;

        v_str[v_idx] = e_str;
        if (cnt_lines % e_line_sz == 0)
        {
            if (filter_out_e(v_str))
            {
                cnt_bad_base++;
            }
            else
            {
                p_s();
            }
        }

    }

    std::cerr << endl << "- cnt_bad_base : " << cnt_bad_base << endl;


    if_.close();
    of_.close();

    clear_range();
#endif 




#if 0
    ifstream if_("txt.txt"); 
    assert(if_.is_open());

    string e_str="NULL"; 
    int cnt = 9;
    while (!if_.eof())
    {
        if_ >> e_str;
        cout << "___"<<e_str << "___" << endl;
    }
    if_.close();
#endif 
    



#if 0

    int arr[1024] = { 0 };
    int cnt = 0;
    for (auto &e : arr)
    {
        e = cnt++;
    }
#define R(N) vector<int>(arr + 0, arr + N)
#define R_(M,N) vector<int>(arr + M, arr + N)


    for (auto i : R(9))
    {
        cout << i << endl; 
    }
    for (auto i : R_(2,9))
    {
        cout << i << endl;
    }
    
#endif



#if 0

    vector<vector<char>> mg =
    {
        { '#', '#', '#', '#', '#', '#', '#', '#' },
        { '#', 'S', ' ', ' ', ' ', ' ', ' ', '#' },
        { '#', '#', '#', ' ', '#', '#', '#', '#' },
        { '#', ' ', '#', ' ', '#', '#', '#', '#' },
        { '#', ' ', ' ', ' ', ' ', ' ', ' ', '#' },
        { '#', ' ', '#', '#', '#', '#', '#', '#' },
        { '#', ' ', ' ', ' ', ' ', ' ', 'F', '#' },
        { '#', '#', '#', '#', '#', '#', '#', '#' }
    };

    auto s = pair<int, int>(1, 1);
    auto f = pair<int, int>(6, 6);

    vector<tuple<int, int,int,int>> visitor{};

    vector<pair<int,int>> D = { {-1,0}, {0,-1}, {0,1}, {1,0} };
    

    auto s_ = pair<int, int>(s);

    
    int cnt = 0;
    int cnt_times = 17;
    while (cnt_times--)
    {
        
        for (int i = cnt; i < D.size()+ cnt; i++)
        {
            auto e = D[i%D.size()];
            auto s_old = pair<int, int>(s_);
            s_.first += e.first;
            s_.second += e.second;

            if (mg[s_.first][s_.second] == ' ')
            {
                auto if_visitor_has_ele = [visitor,s_old,e]() {
                    auto ret_i = 0;
                    for (auto e_v : visitor)
                    {
                        int _0, _1, _2, _3;
                        tie(_0,_1,_2,_3) = e_v;
                        if (_0 == s_old.first && _1 == s_old.second &&
                            _2 == e.first && _2 == e.second)
                        {
                            ret_i = 1;
                            break;
                        }
                        
                    }
                    return ret_i;
                };

                if (! if_visitor_has_ele() )
                {
                    visitor.push_back(make_tuple(s_old.first, s_old.second, e.first, e.second));
                    cout << s_.first << " " << s_.second << endl;
                    i--;
                }
                else // retrate
                {
                    s_ = pair<int, int>(s_old);
                    cnt++;
                }

            }
            else
            {
                s_ = pair<int, int>(s_old);
                break;
            }
        }

        cnt++;
      
    }



    
 


    
    

#endif 
#if 0
    string e_str = "ABCD";
    string e_str_ = std::string(e_str);

    std::transform(e_str.begin(), e_str.end(), e_str_.begin(), [](char e_c) {return e_c + 1; });
    cout << e_str << endl; 
    cout << e_str_ << endl; 
#endif

#if 0

    ifstream if_("E:/jd/t/1.txt");

    assert(if_.is_open()); 
  
    std::stringstream fc_;
    fc_ << if_.rdbuf(); 
    auto &fc = fc_.str();


   cout << "___" << fc << "___" << endl;

   auto id_v = split_str_2_vec(fc, ' ', 3);
   for (auto &e : id_v)
   {
       cout << "___" << e << "___" << endl;
   }
    
    if_.close();


#endif 

#if 0

    vector<int> v0(3, 1); 
    vector<int> v1(19, 22); 
    std::copy(v0.begin(), v0.end(), v1.begin() + 2); 
    std::copy(v0.begin(), v0.end(), (int*)(v1.data()) + 11);

    for (auto &e : v1)
    {
        cout << e << endl; 
    }
    



#endif 

#if 0

    auto f_0 = std::async();




    std::future<int> f1 = std::async(std::launch::async, []() {
        return 8;
    });

    cout << f1.get() << endl; //output: 8

    std::future<int> f2 = std::async(std::launch::async, []() {
        cout << 8 << endl;
        return 8;
    });

    f2.wait(); //output: 8

    std::future<int> future = std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        return 8;
    });

    std::cout << "waiting...\n";
    std::future_status status;



    do {
        status = future.wait_for(std::chrono::seconds(1));
        if (status == std::future_status::deferred) {
            std::cout << "deferred\n";
        }
        else if (status == std::future_status::timeout) {
            std::cout << "timeout\n";
        }
        else if (status == std::future_status::ready) {
            std::cout << "ready!\n";
        }
    } while (status != std::future_status::ready);

    std::cout << "result is " << future.get() << '\n';


   






    


    

#endif 


#if 0

    vector<int> a{ 2 };
    vector<int> b{2,4,6}; 
  

    int cnt = 0;
    

    vector<int> ans;

   
    int j = 0; 
    int i = 0;

    float res = 12345;
    if (a.size() == 0)
    {
        auto &_v = b;
        auto len = _v.size();
        res = len % 2 == 0 ? (_v[len / 2] + _v[len / 2 - 1]) / 2 : _v[len / 2];
    }
 
    else if (b.size() == 0)
    {
        auto &_v = a;
        auto len = _v.size();
        res = len % 2 == 0 ? (_v[len / 2] + _v[len / 2 - 1]) / 2 : _v[len / 2];
    }


    if (res != 12345)
    {
        cout << res <<endl;
    }

    while(1)
    {

        if (cnt * 2 >= a.size() + b.size()) break;

        if (i == a.size())
        {
            ans.push_back(b[j]);
            j++;
            cnt++;
            continue;
        }
 

        if (j == b.size())
        {
            ans.push_back(a[i]);
            i++;
            cnt++;
            continue;
        }
      


        if (i < a.size() && j < b.size() && a[i] <= b[j])
        {
                ans.push_back(a[i]);
                i++;
        }
        else if (j < b.size() && i< a.size() && a[i] > b[j] )
        {
            ans.push_back(b[j]);
            j++;
        }
        cnt++;
    }

    for (auto &e : ans)
    {
        cout << e << " " << endl;
    }

#if 0



    if (ans.size() % 2 == 0)
    {
        res = ans[ans.size() - 1];
    }
    else
    {
         res = ans[ans.size() - 1] + ans[ans.size() - 2];
         res = res / 2.0;

    }

    cout << res; 
#endif
    

    


#endif 


#if 0

   //l1, l2
    //l1_,l2_
    int C = 0;
    auto head = NULL; 
    while (l1_ || l2_)
    {

        auto v1 = 0; 
        auto v2 = 0;

        
        if (! l1_->next)
        {
            v1 = l1_->val;
            l1_ = l1_->next;
        
        }

        if (! l2_->next)
        {
            v2 = l2_->val;
            l2_ = l2_->next;
        }

        auto v = (v1 + v2 + C) % 10;
        C = (v1 + v2 + C >= 10)? 1:0;

        // print v
    }



#endif 

#if 0

    auto lengthOfLongestSubstring = [](string s) {


        map<char, int> map_;

        auto exists_e_c = [&map_](char e_c)
        {
            return map_.find(e_c) == map_.end() ? 0 : 1;
        };

        int max_len = 0;

        for (auto &e_c : s)
        {
            if (exists_e_c(e_c))
            {
                int max_len_ = map_.size();
                max_len = max_len_ > max_len ? max_len_ : max_len;
                map_.clear();
                map_[e_c] = 1;
            }
            else
            {
                map_[e_c] = 1;
            }
        }

        int max_len_ = map_.size();
        max_len = max_len_ > max_len ? max_len_ : max_len;
        return max_len;



    };

    cout << lengthOfLongestSubstring(string("dvdf")) << endl;


#endif






#if 0

    const int N = (const int)(2e1 + 0);
    const int td_num = 2;
    
    vector<float> arr_src{};
    arr_src.resize(N);
    int  cnt = 0;
    for (auto &e : arr_src)
    {
        e = (cnt++) * 1.0f;
    }

    exec_multi_threads_on_data(arr_src, N, td_num, exec_e);





#endif 


#if 0
	typedef  pair<int, std::thread> i_td;

	vector<i_td> arr_pair_i_td{};

	const int N = (const int)(2e2 + 3);
	int mod_num = 3;
	arr_pair_i_td.resize(N % mod_num == 0 ? N / mod_num : N / mod_num + 1);

	int cnt = 0;
	int cnt_arr = 0;


	vector<float> arr_src{};
	arr_src.resize(N);
	cnt = 0;
	for (auto &e : arr_src)
	{
		e = (cnt++) * 1.0f;
	}

    cnt = 0;
	while (cnt < N)
	{

		//cout << cnt << endl;

		int id_td = (int)(cnt / mod_num);
        cout << id_td << endl; 
		auto& pair_i_i = make_pair(cnt, cnt + mod_num > N ? N : cnt + mod_num);


        arr_pair_i_td[id_td].first = id_td;
        arr_pair_i_td[id_td].second = thread([](int id_td, const vector<float>& arr_src, pair<int, int> pair_i_i) {
           
#if 1
            if (id_td == 0)
            {
                system("sleep 0s");
            }
            else if(id_td == 1)
            {
                system("sleep 0.1s");
            }
            else if (id_td == 2)
            {
                system("sleep 0.2s");
            }
            else if (id_td == 3)
            {
                system("sleep 0.3s");
            }
            else if (id_td == 4)
            {
                system("sleep 0.4s");
            }
            else
            {
                system("sleep 0.6s");
            }

#endif 
            cout << "- id_td: " << id_td << endl;
            //cout << pair_i_i.first << " " << pair_i_i.second << endl; 
            for (auto i = pair_i_i.first; i < pair_i_i.second; i++)
            {
                cout << arr_src[i] << " ";
            }
            cout << endl << "--------------------" << endl; 

        },
            id_td, std::ref(arr_src), pair_i_i
            );

            

		//cnt_arr++;
		cnt += mod_num;


        //arr_pair_i_td[id_td].second.join();
	}


	for (auto& e_pair : arr_pair_i_td)
	{
		//cout << e_pair.first << endl; 
		e_pair.second.join();
	}



#endif

#if 0

	typedef  pair<std::thread, std::pair<int, int>> i__pair_i_i;

	vector<i__pair_i_i> arr_pair_i_i{};


#if 1
	const int N = 2e2 + 3;
	int mod_num = 200;
	arr_pair_i_i.resize( N % mod_num == 0? N/mod_num : N/mod_num + 1);

	int cnt = 0;
	int cnt_arr = 0;





	while (cnt < N)
	{

		//cout << cnt << endl;
		auto& e_pair = make_pair(
				std::thread([]() {cout << 1 << endl; }),
				std::make_pair(cnt, cnt + mod_num > N ? N : cnt + mod_num)
				);
		arr_pair_i_i[cnt / mod_num] = e_pair;
		//cnt_arr++;
		cnt += mod_num;
	}

	// cout << cnt << endl; 

	for (auto &e_pair : arr_pair_i_i)
	{
		e_pair.first.join(); 
	}


	vector<float> arr_src{};
	arr_src.resize(N); 
	cnt = 0; 
	for (auto &e : arr_src)
	{
		e = cnt++;
	}

#if 0
	int id_td = 0; 
	for (auto &e_args : arr_pair_i_i)
	{

		auto e_td = std::thread(
				/* exec_e_td() */
				[](vector<float> &arr_src, vector<std::pair<int, int> >& arr_pair_i_i, int& id_td) {

				auto &e_pair_i_i = arr_pair_i_i[id_td];

				cout << "- id_td: " << id_td << endl; 
				system("sleep 1s");
				for (int i = e_pair_i_i.first; i < e_pair_i_i.second; i++)
				{
				cout << arr_src[i] << " ";
				}
				cout << endl;
				},
				std::ref(arr_src),
				std::ref(arr_pair_i_i),
				std::ref(id_td)
				);

		e_td.join();
		id_td++;
	}
#endif 



	//cnt = 2;
	//while (cnt--)
	{
		//system("sleep 2s");
	}
#endif 

#endif 



#if 0

	auto g_cnt = 714; 
	auto td_num = 4; 
	auto t = ceil(g_cnt * 1.0f / td_num);       // 179
	// t = ceil(g_cnt / td_num);                // => 178

	cout << t; 

	auto id_td_args_0 = td_args(2); 
	auto id_td_args_1 = td_args(2<<1);


	auto tid_0 = thread(td_exec_0, std::ref(id_td_args_0));
	auto tid_1 = thread(td_exec_1, std::ref(id_td_args_1));

	tid_0.join(); 
	tid_1.join();

	//id_td_args_0.~td_args();
	//id_td_args_1.~td_args();


	cout << "- end" << endl; 
#endif 



#if 0
	auto *ACGT = "ACGT";

	int arr[] = { 3, 3, 1, 3, 3, 2, 1, 3, 2, 0, 1 };

	for (auto &e : arr)
	{
		cout << ACGT[e];
	}
#endif




	__P__;
}
