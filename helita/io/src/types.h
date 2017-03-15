#ifndef __TYPES_H__  // __TYPES_H__
#define __TYPES_H__

//#define _64_BIT_OS_
// Todo: 
// byte -> uint8_t
// int08-> int8
// others: stdtypes, include in configure
// test for idl 64bit or not -> shell script
// make makefile with manual target for libraries

/*
Superseded by inttypes.h
typedef unsigned char byte;
typedef signed char int08_t;
typedef short int int16_t;
typedef int int32_t;
//#ifdef _64_BIT_OS_
// TvW: why define this while including types.h? Only problem with 64 bit?
//typedef long int int64_t;
//#else
typedef long long int int64_t;
//#endif
*/
typedef float float32_t;
typedef double float64_t;

typedef float64_t fp_t;  // the default floating point type

struct complex{
  fp_t re;
  fp_t im;
};

typedef struct complex complex_t;
struct fzhead {                    // first block for fz files
  int synch_pattern;
  uint8_t subf;
  uint8_t source;
  uint8_t nhb,datyp,ndim,file_class;
  uint8_t cbytes[4];	      // can't do as int because of %4 rule
  uint8_t free[178];
  int dim[16];
  char txt[256];
};

struct compresshead{
  int tsize,nblocks,bsize;
  uint8_t slice_size,type;
};

void bswapi16(int16_t *x,int n);
void bswapi32(int32_t *x,int n);
void bswapi64(int64_t *x,int n);

#endif               // __TYPES_H__
