// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
#include "arrow/dataset/file_rados_parquet.h"

#include "arrow/api.h"
#include "arrow/dataset/dataset_internal.h"
#include "arrow/dataset/expression.h"
#include "arrow/dataset/file_base.h"
#include "arrow/filesystem/filesystem.h"
#include "arrow/filesystem/path_util.h"
#include "arrow/filesystem/util_internal.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/iterator.h"
#include "arrow/util/logging.h"
#include "arrow/io/api.h"

namespace arrow {
namespace dataset {

class RandomAccessObject : public arrow::io::RandomAccessFile {
 public:
  explicit RandomAccessObject(std::string oid, 
                              std::shared_ptr<RadosCluster> cluster)
    : oid_(oid),
      cluster_(std::move(cluster)) {}

  arrow::Status Init() {
    uint64_t size;
    int e = cls_cxx_stat(hctx_, &size, NULL);
    if (e == 0) {
      content_length_ = size;
      return arrow::Status::OK();
    } else {
      return arrow::Status::ExecutionError("cls_cxx_stat returned non-zero exit code.");
    }
  }

  arrow::Status CheckClosed() const {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed stream");
    }
    return arrow::Status::OK();
  }

  arrow::Status CheckPosition(int64_t position, const char* action) const {
    if (position < 0) {
      return arrow::Status::Invalid("Cannot ", action, " from negative position");
    }
    if (position > content_length_) {
      return arrow::Status::IOError("Cannot ", action, " past end of file");
    }
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) { return 0; }

  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) {
    RETURN_NOT_OK(CheckClosed());
    RETURN_NOT_OK(CheckPosition(position, "read"));

    // No need to allocate more than the remaining number of bytes
    nbytes = std::min(nbytes, content_length_ - position);

    if (nbytes > 0) {
      ceph::bufferlist* bl = new ceph::bufferlist();
      cluster_->ioCtx->read(oid_.c_str(), bl, nbytes, position);
      return std::make_shared<arrow::Buffer>((uint8_t*)bl->c_str(), bl->length());
    }
    return std::make_shared<arrow::Buffer>("");
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) {
    ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
    pos_ += buffer->size();
    return std::move(buffer);
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) {
    ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(pos_, nbytes, out));
    pos_ += bytes_read;
    return bytes_read;
  }

  arrow::Result<int64_t> GetSize() {
    RETURN_NOT_OK(CheckClosed());
    return content_length_;
  }

  arrow::Status Seek(int64_t position) {
    RETURN_NOT_OK(CheckClosed());
    RETURN_NOT_OK(CheckPosition(position, "seek"));

    pos_ = position;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const {
    RETURN_NOT_OK(CheckClosed());
    return pos_;
  }

  arrow::Status Close() {
    closed_ = true;
    return arrow::Status::OK();
  }

  bool closed() const { return closed_; }

 protected:
  std::string oid_;
  std::shared_ptr<RadosCluster> cluster_;
  bool closed_ = false;
  int64_t pos_ = 0;
  int64_t content_length_ = -1;
};

class RadosParquetScanTask : public ScanTask {
 public:
  RadosParquetScanTask(std::shared_ptr<ScanOptions> options,
                       std::shared_ptr<ScanContext> context, FileSource source,
                       std::shared_ptr<DirectObjectAccess> doa)
      : ScanTask(std::move(options), std::move(context)),
        source_(std::move(source)),
        doa_(std::move(doa)) {}

  Result<RecordBatchIterator> Execute() override {

    auto oid = doa_->ConvertFileNameToObjectID(source_.path());

    auto file = std::make_shared<RandomAccessObject>(oid, doa_->cluster());
    ARROW_RETURN_NOT_OK(file->Init());
    arrow::dataset::FileSource source(file);
    auto format = std::make_shared<arrow::dataset::ParquetFileFormat>();
    ARROW_ASSIGN_OR_RAISE(auto fragment,
                          format->MakeFragment(source, options_->partition_expression));
    auto ctx = std::make_shared<arrow::dataset::ScanContext>();
    auto builder =
        std::make_shared<arrow::dataset::ScannerBuilder>(options_->dataset_schema, fragment, ctx);
    ARROW_RETURN_NOT_OK(builder->Filter(options_->filter));
    ARROW_RETURN_NOT_OK(builder->Project( options_->projector.schema()->field_names()));

    ARROW_ASSIGN_OR_RAISE(auto scanner, builder->Finish());
    ARROW_ASSIGN_OR_RAISE(auto table, scanner->ToTable());

    ARROW_RETURN_NOT_OK(file->Close());

    auto table_reader = std::make_shared<arrow::TableBatchReader>(*table);
    RecordBatchVector batches;
    table_reader->ReadAll(&batches);
    return MakeVectorIterator(batches);
  }

 protected:
  FileSource source_;
  std::shared_ptr<DirectObjectAccess> doa_;
};

RadosParquetFileFormat::RadosParquetFileFormat(const std::string& ceph_config_path,
                                               const std::string& data_pool,
                                               const std::string& user_name,
                                               const std::string& cluster_name) {
  auto cluster = std::make_shared<RadosCluster>(ceph_config_path, data_pool, user_name,
                                                cluster_name);
  cluster->Connect();
  auto doa = std::make_shared<arrow::dataset::DirectObjectAccess>(cluster);
  doa_ = doa;
}

Result<std::shared_ptr<Schema>> RadosParquetFileFormat::Inspect(
    const FileSource& source) const {
  ceph::bufferlist in, out;

  Status s = doa_->Exec(source.path(), "read_schema", in, out);
  if (!s.ok()) return Status::ExecutionError(s.message());

  std::vector<std::shared_ptr<Schema>> schemas;
  ipc::DictionaryMemo empty_memo;
  io::BufferReader schema_reader((uint8_t*)out.c_str(), out.length());
  ARROW_ASSIGN_OR_RAISE(auto schema, ipc::ReadSchema(&schema_reader, &empty_memo));
  return schema;
}

Result<ScanTaskIterator> RadosParquetFileFormat::ScanFile(
    std::shared_ptr<ScanOptions> options, std::shared_ptr<ScanContext> context,
    FileFragment* file) const {
  std::shared_ptr<ScanOptions> options_ = std::make_shared<ScanOptions>(*options);
  options_->partition_expression = file->partition_expression();
  options_->dataset_schema = file->dataset_schema();
  options_->bypass_fap_scantask = false;
  ScanTaskVector v{std::make_shared<RadosParquetScanTask>(
      std::move(options_), std::move(context), file->source(), std::move(doa_))};
  return MakeVectorIterator(v);
}

}  // namespace dataset
}  // namespace arrow
