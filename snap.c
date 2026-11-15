#include <string.h>
#include <stddef.h>
long snap(const unsigned char*s,size_t n,unsigned char*o,size_t on){
  size_t i=0,k=0; while(s[i]&0x80)i++; i++;
  while(i<n){
    unsigned c=s[i++]; unsigned t=c&3; size_t len,off;
    if(t==0){ len=c>>2; if(len>=60){unsigned nb=len-59; len=0; for(unsigned j=0;j<nb;j++) len|=(size_t)s[i+j]<<(8*j); i+=nb;} len+=1;
      if(k+len>on||i+len>n) return -1; memcpy(o+k,s+i,len); i+=len;k+=len; continue;}
    if(t==1){ len=((c>>2)&7)+4; off=((size_t)(c>>5)<<8)|s[i]; i+=1;}
    else if(t==2){ len=(c>>2)+1; off=s[i]|((size_t)s[i+1]<<8); i+=2;}
    else { len=(c>>2)+1; off=s[i]|((size_t)s[i+1]<<8)|((size_t)s[i+2]<<16)|((size_t)s[i+3]<<24); i+=4;}
    if(off==0||off>k||k+len>on) return -2;
    for(size_t j=0;j<len;j++) o[k+j]=o[k+j-off]; k+=len;
  }
  return (long)k;
}
