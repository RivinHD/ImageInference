// SPDX-FileCopyrightText: © 2024 Vincent Gerlach
//
// SPDX-License-Identifier: MIT

#ifndef IMAGEINFERENCE_MACROS_H
#define IMAGEINFERENCE_MACROS_H

#define CACHE_LINE_SIZE 64
#define PAGE_CACHE_ALIGN(type, size) (sizeof(type) * (size) >= 4096 ? 4096 : (sizeof(type) * (size) >= CACHE_LINE_SIZE ? CACHE_LINE_SIZE : sizeof(type)))

#endif // IMAGEINFERENCE_MACROS_H