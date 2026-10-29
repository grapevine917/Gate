#include "logger.h"
#include "disk_utils.h"
#include "partition.h"
namespace diskann {
int recovery_merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                 const std::string &idmaps_suffix, const uint64_t nshards, uint32_t max_degree,
                 const std::string &output_vamana, const std::string &medoids_file, bool use_filters,
                 const std::string &labels_to_medoids_file)
{
    // Read ID maps
    std::vector<std::string> vamana_names(nshards);
    std::vector<std::vector<uint32_t>> idmaps(nshards);
    for (uint64_t shard = 0; shard < nshards; shard++)
    {
        vamana_names[shard] = vamana_prefix + std::to_string(shard) + vamana_suffix;
        read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix, idmaps[shard]);
    }

    // find max node id
    size_t nnodes = 0;
    size_t nelems = 0;
    for (auto &idmap : idmaps)
    {
        for (auto &id : idmap)
        {
            nnodes = std::max(nnodes, (size_t)id);
        }
        nelems += idmap.size();
    }
    nnodes++;
    diskann::cout << "# nodes: " << nnodes << ", max. degree: " << max_degree << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<uint32_t, uint32_t>> node_shard;
    node_shard.reserve(nelems);
    for (size_t shard = 0; shard < nshards; shard++)
    {
        diskann::cout << "Creating inverse map -- shard #" << shard << std::endl;
        for (size_t idx = 0; idx < idmaps[shard].size(); idx++)
        {
            size_t node_id = idmaps[shard][idx];
            node_shard.push_back(std::make_pair((uint32_t)node_id, (uint32_t)shard));
        }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
        return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    diskann::cout << "Finished computing node -> shards map" << std::endl;

    // will merge all the labels to medoids files of each shard into one
    // combined file
    if (use_filters)
    {
        std::unordered_map<uint32_t, std::vector<uint32_t>> global_label_to_medoids;

        for (size_t i = 0; i < nshards; i++)
        {
            std::ifstream mapping_reader;
            std::string map_file = vamana_names[i] + "_labels_to_medoids.txt";
            mapping_reader.open(map_file);

            std::string line, token;
            uint32_t line_cnt = 0;

            while (std::getline(mapping_reader, line))
            {
                std::istringstream iss(line);
                uint32_t cnt = 0;
                uint32_t medoid = 0;
                uint32_t label = 0;
                while (std::getline(iss, token, ','))
                {
                    token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                    token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());

                    uint32_t token_as_num = std::stoul(token);

                    if (cnt == 0)
                        label = token_as_num;
                    else
                        medoid = token_as_num;
                    cnt++;
                }
                global_label_to_medoids[label].push_back(idmaps[i][medoid]);
                line_cnt++;
            }
            mapping_reader.close();
        }

        std::ofstream mapping_writer(labels_to_medoids_file);
        assert(mapping_writer.is_open());
        for (auto iter : global_label_to_medoids)
        {
            mapping_writer << iter.first << ", ";
            auto &vec = iter.second;
            for (uint32_t idx = 0; idx < vec.size() - 1; idx++)
            {
                mapping_writer << vec[idx] << ", ";
            }
            mapping_writer << vec[vec.size() - 1] << std::endl;
        }
        mapping_writer.close();
    }

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(nshards);
    for (size_t i = 0; i < nshards; i++)
    {
        vamana_readers[i].open(vamana_names[i], (16ULL * 1024 * 1024));
        size_t expected_file_size;
        vamana_readers[i].read((char *)&expected_file_size, sizeof(uint64_t));
    }

    size_t vamana_metadata_size =
        sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t); // expected file size + max degree +
                                                                                   // medoid_id + frozen_point info

    // create cached vamana writers
    cached_ofstream merged_vamana_writer(output_vamana, (16ULL * 1024 * 1024));

    size_t merged_index_size = vamana_metadata_size; // we initialize the size of the merged index to
                                                     // the metadata size
    size_t merged_index_frozen = 0;
    merged_vamana_writer.write((char *)&merged_index_size,
                               sizeof(uint64_t)); // we will overwrite the index size at the end

    uint32_t output_width = max_degree;
    uint32_t max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(uint32_t) bytes
    for (auto &reader : vamana_readers)
    {
        uint32_t input_width;
        reader.read((char *)&input_width, sizeof(uint32_t));
        max_input_width = input_width > max_input_width ? input_width : max_input_width;
    }

    diskann::cout << "Max input width: " << max_input_width << ", output width: " << output_width << std::endl;

    merged_vamana_writer.write((char *)&output_width, sizeof(uint32_t));
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    uint32_t nshards_u32 = (uint32_t)nshards;
    uint32_t one_val = 1;
    medoid_writer.write((char *)&nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *)&one_val, sizeof(uint32_t));

    uint64_t vamana_index_frozen = 0; // as of now the functionality to merge many overlapping vamana
                                      // indices is supported only for bulk indices without frozen point.
                                      // Hence the final index will also not have any frozen points.
    for (uint64_t shard = 0; shard < nshards; shard++)
    {
        uint32_t medoid;
        // read medoid
        vamana_readers[shard].read((char *)&medoid, sizeof(uint32_t));
        vamana_readers[shard].read((char *)&vamana_index_frozen, sizeof(uint64_t));
        assert(vamana_index_frozen == false);
        // rename medoid
        medoid = idmaps[shard][medoid];

        medoid_writer.write((char *)&medoid, sizeof(uint32_t));
        // write renamed medoid
        if (shard == (nshards - 1)) //--> uncomment if running hierarchical
            merged_vamana_writer.write((char *)&medoid, sizeof(uint32_t));
    }
    merged_vamana_writer.write((char *)&merged_index_frozen, sizeof(uint64_t));
    medoid_writer.close();

    diskann::cout << "Starting merge" << std::endl;

    // Gopal. random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<uint32_t> final_nhood;

    uint32_t nnbrs = 0, shard_nnbrs = 0;
    uint32_t cur_id = 0;
    for (const auto &id_shard : node_shard)
    {
        uint32_t node_id = id_shard.first;
        uint32_t shard_id = id_shard.second;
        if (cur_id < node_id)
        {
            // Gopal. random_shuffle() is deprecated.
            std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
            nnbrs = (uint32_t)(std::min)(final_nhood.size(), (uint64_t)max_degree);
            // write into merged ofstream
            merged_vamana_writer.write((char *)&nnbrs, sizeof(uint32_t));
            merged_vamana_writer.write((char *)final_nhood.data(), nnbrs * sizeof(uint32_t));
            merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
            if (cur_id % 499999 == 1)
            {
                diskann::cout << "." << std::flush;
            }
            cur_id = node_id;
            nnbrs = 0;
            for (auto &p : final_nhood)
                nhood_set[p] = 0;
            final_nhood.clear();
        }
        // read from shard_id ifstream
        vamana_readers[shard_id].read((char *)&shard_nnbrs, sizeof(uint32_t));

        if (shard_nnbrs == 0)
        {
            diskann::cout << "WARNING: shard #" << shard_id << ", node_id " << node_id << " has 0 nbrs" << std::endl;
        }

        std::vector<uint32_t> shard_nhood(shard_nnbrs);
        if (shard_nnbrs > 0)
            vamana_readers[shard_id].read((char *)shard_nhood.data(), shard_nnbrs * sizeof(uint32_t));
        // rename nodes
        for (uint64_t j = 0; j < shard_nnbrs; j++)
        {
            if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0)
            {
                nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
                final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
            }
        }
    }

    // Gopal. random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (uint32_t)(std::min)(final_nhood.size(), (uint64_t)max_degree);
    // write into merged ofstream
    merged_vamana_writer.write((char *)&nnbrs, sizeof(uint32_t));
    if (nnbrs > 0)
    {
        merged_vamana_writer.write((char *)final_nhood.data(), nnbrs * sizeof(uint32_t));
    }
    merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
    for (auto &p : final_nhood)
        nhood_set[p] = 0;
    final_nhood.clear();

    diskann::cout << "Expected size: " << merged_index_size << std::endl;

    merged_vamana_writer.reset();
    merged_vamana_writer.write((char *)&merged_index_size, sizeof(uint64_t));

    diskann::cout << "Finished merge" << std::endl;
    return 0;
}

}
int main(int argc,char**argv) {
 if(argc!=3)return 2;
 std::string p=argv[1], base=argv[2];
 diskann::recovery_merge_shards(p+"_mem.index_tempFiles_subshard-","_mem.index",p+"_mem.index_tempFiles_subshard-","_ids_uint32.bin",17,64,p+"_mem.index",p+"_disk.index_medoids.bin",false,"");
 diskann::create_disk_layout<uint8_t>(base,p+"_mem.index",p+"_disk.index");
 gen_random_slice<uint8_t>(base,p+"_sample",0.001);
 return 0;
}
