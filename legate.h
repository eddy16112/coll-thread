#pragma once

#define LEGATE_ABORT                                                                        \
  do {                                                                                      \
    log_coll.error(                                                               \
      "Legate called abort in %s at line %d in function %s", __FILE__, __LINE__, __func__); \
    abort();                                                                                \
  } while (false)
