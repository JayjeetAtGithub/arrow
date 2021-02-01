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
#define _FILE_OFFSET_BITS 64

#include <iostream>
#include <random>
#include <rados/objclass.h>
#include <rados/librados.hpp>
#include "arrow/api.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/dataset_rados.h"
#include "arrow/dataset/rados_utils.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"
#include "arrow/util/iterator.h"
#include "arrow/adapters/arrow-rados-cls/cls_arrow_test_utils.h"
#include "gtest/gtest.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"

using arrow::dataset::string_literals::operator"" _;

std::shared_ptr<arrow::dataset::RadosCluster> CreateTestClusterHandle() {
  auto cluster = std::make_shared<arrow::dataset::RadosCluster>("cephfs_data",
                                                                "/etc/ceph/ceph.conf");
  cluster->Connect();
  return cluster;
}

arrow::dataset::RadosDatasetFactoryOptions CreateTestRadosFactoryOptions() {
  auto cluster = CreateTestClusterHandle();
  arrow::dataset::RadosDatasetFactoryOptions factory_options;
  factory_options.ceph_config_path = cluster->ceph_config_path;
  factory_options.cls_name = cluster->cls_name;
  factory_options.cluster_name = cluster->cluster_name;
  factory_options.flags = cluster->flags;
  factory_options.pool_name = cluster->pool_name;
  factory_options.user_name = cluster->user_name;
  return factory_options;
}

std::shared_ptr<arrow::dataset::RadosFileSystem> CreateTestRadosFileSystem() {
  auto cluster = CreateTestClusterHandle();
  auto fs = std::make_shared<arrow::dataset::RadosFileSystem>();
  arrow::Status s = fs->Init(cluster);
  if (!s.ok()) std::cout << "Init() failed.\n";
  return fs;
}

double RandDouble(double min, double max) {
  return min + ((double)rand() / RAND_MAX) * (max - min);
}

int32_t RandInt32(int32_t min, int32_t max) {
  return min + (rand() % static_cast<int>(max - min + 1));
}

std::shared_ptr<arrow::Table> CreatePartitionedTable() {
  arrow::MemoryPool* pool = arrow::default_memory_pool();

  arrow::Int32Builder sales_builder(pool);
  for (int i = 0; i < 10; i++) {
    sales_builder.Append(RandInt32(800, 1000));
  }
  std::shared_ptr<arrow::Int32Array> sales_array;
  sales_builder.Finish(&sales_array);

  arrow::DoubleBuilder price_builder(pool);
  for (int i = 0; i < 10; i++) {
    price_builder.Append(RandDouble(38999.56, 99899.23));
  }
  std::shared_ptr<arrow::DoubleArray> price_array;
  price_builder.Finish(&price_array);

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("sales", arrow::int32()), 
      arrow::field("price", arrow::float64())
  };
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  return arrow::Table::Make(schema, {sales_array, price_array});
}

TEST(TestClsSDK, EndToEndWithoutPartitionPruning) {
  auto fs = CreateTestRadosFileSystem();
  auto factory_options = CreateTestRadosFactoryOptions();
  factory_options.partition_base_dir = "/tesla";

  auto writer = std::make_shared<arrow::dataset::CephFSParquetWriter>(fs);
  writer->WriteTable(CreatePartitionedTable(), "/tesla/18UK.parquet");
  writer->WriteTable(CreatePartitionedTable(), "/tesla/18US.parquet");
  writer->WriteTable(CreatePartitionedTable(), "/tesla/19UK.parquet");
  writer->WriteTable(CreatePartitionedTable(), "/tesla/19US.parquet");
  writer->WriteTable(CreatePartitionedTable(), "/tesla/20UK.parquet");
  writer->WriteTable(CreatePartitionedTable(), "/tesla/20US.parquet");

  arrow::dataset::FinishOptions finish_options;
  auto factory = arrow::dataset::RadosDatasetFactory::Make(fs, factory_options).ValueOrDie();
  auto ds = factory->Finish(finish_options).ValueOrDie();

  auto builder = ds->NewScan().ValueOrDie();
  auto projection = std::vector<std::string>{"price", "sales"};
  auto filter = ("sales"_ > int32_t(900) && "price"_ > double(90000.0f)).Copy();

  builder->Project(projection);
  builder->Filter(filter);
  auto scanner = builder->Finish().ValueOrDie();

  auto table = scanner->ToTable().ValueOrDie();
  std::cout << table->ToString() << "\n";
  std::cout << table->num_rows() << "\n";
}

TEST(TestClsSDK, EndToEndWithPartitionPruning) {
  auto fs = CreateTestRadosFileSystem();
  auto factory_options = CreateTestRadosFactoryOptions();
  factory_options.partition_base_dir = "/tesla";

  auto writer = std::make_shared<arrow::dataset::CephFSParquetWriter>(fs);
  writer->WriteTable(CreatePartitionedTable(), "/tesla/year=2018/country=UK/18UK.parquet");
  writer->WriteTable(CreatePartitionedTable(), "/tesla/year=2018/country=US/18US.parquet");
  writer->WriteTable(CreatePartitionedTable(), "/tesla/year=2019/country=UK/19UK.parquet");
  writer->WriteTable(CreatePartitionedTable(), "/tesla/year=2019/country=US/19US.parquet");
  writer->WriteTable(CreatePartitionedTable(), "/tesla/year=2020/country=UK/20UK.parquet");
  writer->WriteTable(CreatePartitionedTable(), "/tesla/year=2020/country=US/20US.parquet");

  factory_options.partitioning = std::make_shared<arrow::dataset::HivePartitioning>(
      arrow::schema({arrow::field("year", arrow::int32()), arrow::field("country", arrow::utf8())}));

  arrow::dataset::FinishOptions finish_options;
  auto factory = arrow::dataset::RadosDatasetFactory::Make(fs, factory_options).ValueOrDie();
  auto ds = factory->Finish(finish_options).ValueOrDie();

  auto builder = ds->NewScan().ValueOrDie();
  auto projection = std::vector<std::string>{"year", "price", "country", "sales"};
  auto filter = ("sales"_ > int32_t(900) && "price"_ > double(90000.0f) && "year"_ == 2018).Copy();

  builder->Project(projection);
  builder->Filter(filter);
  auto scanner = builder->Finish().ValueOrDie();

  auto table = scanner->ToTable().ValueOrDie();
  std::cout << table->ToString() << "\n";
  std::cout << table->num_rows() << "\n";
}
