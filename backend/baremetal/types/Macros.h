#ifndef IMAGEINFERENCE_MACROS_H
#define IMAGEINFERENCE_MACROS_H

#define CACHE_LINE_SIZE 64
#define PAGE_CACHE_ALIGN(type, size) alignas(type * size >= 4096 ? 4096 : (type *size >= CACHE_LINE_SIZE ? CACHE_LINE_SIZE : type))
#define CACHE_ALIGN(type, size) alignas(type * size >= CACHE_LINE_SIZE ? CACHE_LINE_SIZE : type)

#endif // IMAGEINFERENCE_MACROS_H