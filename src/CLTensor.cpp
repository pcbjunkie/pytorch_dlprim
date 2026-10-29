#include "CLTensor.h"
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <deque>
#include <chrono>
#ifdef DLPRIM_USE_CL1_HPP
#error "DLPrimitives need to be compiled agaist cl2.hpp in order to work with pytorch. cl.hpp is not supported and known to fail"
#endif


namespace ptdlprim {
    std::uint64_t CLCache::round(uint64_t v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        v++;
        return v;
    }

    std::unique_ptr<CLMemAllocation> CLCache::allocate(int id,cl::Context &ctx,int64_t orig_size)
    {
        if(allocated_size == 0 && debug_allocator)
            setlocale(LC_ALL,"");

        std::unique_lock<std::mutex> g(lock);

        std::int64_t size = round(std::max(int64_t(1),orig_size));
        // Cache bins help reuse, but no-cache mode cannot reuse those rounded
        // blocks. Avoid doubling large activations just above a power of two.
        char const *tight=std::getenv("OPENCL_TIGHT_ALLOC");
        char const *no_cache=std::getenv("OPENCL_NO_MEM_CACHE");
        if(tight && std::strcmp(tight,"1")==0 && no_cache && std::atoi(no_cache)!=0 && orig_size>=4096) {
            TORCH_CHECK(orig_size<=std::numeric_limits<int64_t>::max()-4095,"OpenCL allocation size overflow");
            size=(orig_size+4095)&~int64_t(4095);
        }
        std::unique_ptr<CLMemAllocation> res;
        
        auto p=allocation.find(size);
        if(reuse_oversized_chunks) {
            int times=0;
            while(p!=allocation.end()) {
                if(!p->second.empty() || times==2) {
                    break;
                }
                ++p;
                times++;
            }
        }

        if(p==allocation.end() || p->second.empty()) {
            res.reset(new CLMemAllocation(id,ctx,size,orig_size));
            allocated_size += res->size;
        }
        else {
            res = std::move(p->second.back());
            TORCH_CHECK(res->size >= orig_size,"Internal validation");
            res->orig_size = orig_size;
            cached_size -= res->size;
            p->second.pop_back();
        }
        requested_size += res->orig_size;
        peak_requested_size = std::max(requested_size,peak_requested_size);
        peak_reserved_size = std::max(allocated_size,peak_reserved_size);
        if(debug_allocator)
            printf("malloc: allocated: %'16ld  requested %'16ld peak-req %'16ld cached %'16ld\n",allocated_size,requested_size,peak_requested_size,cached_size);
        return res;
    }
    void CLCache::release(std::unique_ptr<CLMemAllocation> &&mem,bool retain,bool account_requested)
    {
        std::unique_lock<std::mutex> g(lock);

        int64_t size = mem->size;
        if(account_requested)requested_size -= mem->orig_size;
        if(!retain) {allocated_size -= mem->size;return;}
        cached_size += mem->size;
        if(debug_allocator)
            printf("free  : allocated: %'16ld  requested %'16ld peak-req %'16ld cached %'16ld\n",allocated_size,requested_size,peak_requested_size,cached_size);
        allocation[size].push_back(std::move(mem));
    }
    
    void CLCache::clear()
    {
        std::unique_lock<std::mutex> g(lock);
        {
            allocation_type tmp;
            tmp.swap(allocation);
            allocated_size -= cached_size;
            cached_size = 0;
        }
    }
    std::map<std::string,int64_t> CLCache::memory_stats(bool reset)
    {
        std::unique_lock<std::mutex> g(lock);
        if(reset) {peak_requested_size=requested_size;peak_reserved_size=allocated_size;}
        return {{"allocated_bytes.all.current",requested_size},
                {"allocated_bytes.all.peak",peak_requested_size},
                {"reserved_bytes.all.current",allocated_size},
                {"reserved_bytes.all.peak",peak_reserved_size},
                {"cached_bytes.all.current",cached_size}};
    }
    void CLCache::prepare(dlprim::Context &ctx)
    {
        int64_t mem_size = ctx.device().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        int64_t rounded_mem_size = round(mem_size);
        for(int64_t size=1;size<=rounded_mem_size;size*=2) {
            allocation[size]; // create empty list
        }
        if(debug_allocator) {
            setlocale(LC_ALL,"");
            printf("GPU max memory allocation size %'15ld creating tables up to %'ld\n",mem_size,rounded_mem_size);
        }
    }

    // A single in-order queue owns the command order. Retired allocations are
    // sealed behind a marker submitted AFTER their last possible queued use.
    // No allocation is reused or returned to Clover before that marker completes.
    struct DeferredReleaseState {
        struct Batch {
            cl::Event complete;
            std::vector<std::unique_ptr<CLMemAllocation>> allocations;
        };
        std::mutex lock;
        std::vector<std::unique_ptr<CLMemAllocation>> staged;
        std::deque<Batch> batches;
        bool enabled=false;
        bool timing=false;
        int64_t byte_limit=32*1024*1024;
        unsigned op_limit=8;
        unsigned ops=0;
        int64_t bytes=0,peak_bytes=0,retired_count=0;
        int64_t sync_calls=0,finish_calls=0,host_wait_ns=0,markers=0,flushes=0,throttles=0;

        DeferredReleaseState(bool no_cache) {
            auto flag=std::getenv("OPENCL_DEFERRED_RELEASE");
            enabled=no_cache && flag && std::atoi(flag)!=0;
            flag=std::getenv("OPENCL_SYNC_TIMING");timing=flag && std::atoi(flag)!=0;
            flag=std::getenv("OPENCL_DEFERRED_RELEASE_MB");
            if(flag){int v=std::atoi(flag);TORCH_CHECK(v>=1 && v<=256,"Deferred release budget must be 1..256 MiB");byte_limit=int64_t(v)*1024*1024;}
            flag=std::getenv("OPENCL_ASYNC_OP_LIMIT");
            if(flag){int v=std::atoi(flag);TORCH_CHECK(v>=1 && v<=128,"Async operation limit must be 1..128");op_limit=v;}
        }
        void seal(dlprim::ExecutionContext &q) {
            if(staged.empty())return;
            Batch b;
            q.queue().enqueueMarkerWithWaitList(nullptr,&b.complete);
            b.allocations.swap(staged);
            batches.push_back(std::move(b));++markers;
            q.queue().flush();++flushes;
        }
        void collect(CLCache &cache,bool wait) {
            while(!batches.empty()) {
                auto &b=batches.front();
                if(wait) {
                    auto start=std::chrono::steady_clock::now();
                    b.complete.wait();
                    if(timing)host_wait_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
                }
                else {
                    auto status=b.complete.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
                    TORCH_CHECK(status>=0,"Deferred release marker failed: ",status);
                    if(status!=CL_COMPLETE)break;
                }
                for(auto &mem:b.allocations) {
                    bytes-=mem->size;
                    cache.release(std::move(mem),false,false);
                }
                batches.pop_front();
            }
        }
        void throttle(dlprim::ExecutionContext &q,CLCache &cache) {
            seal(q);
            // Bound both retained memory and queued operations. This is a short
            // pipeline, not unbounded batching or a return to allocation caching.
            auto start=std::chrono::steady_clock::now();
            q.queue().finish();++finish_calls;++throttles;
            if(timing)host_wait_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
            collect(cache,false);ops=0;
        }
    };

    DeferredReleaseState &CLContextManager::deferred(DevData &data)
    {
        static std::mutex initialization_lock;
        std::lock_guard<std::mutex> guard(initialization_lock);
        if(!data.deferred)data.deferred=std::make_shared<DeferredReleaseState>(no_cache_);
        return *data.deferred;
    }

    std::unique_ptr<CLMemAllocation> CLContextManager::alloc(int id,int64_t size)
    {
        auto &inst=instance();auto &d=inst.data(id);auto &s=inst.deferred(d);
        if(s.enabled) {
            std::unique_lock<std::mutex> guard(s.lock);
            s.seal(d.queue);s.collect(d.cache,false);
            if(s.bytes>=s.byte_limit)s.throttle(d.queue,d.cache);
        }
        return d.cache.allocate(id,d.ctx.context(),size);
    }

    void CLContextManager::release(std::unique_ptr<CLMemAllocation> &&mem)
    {
        auto &inst=instance();auto &d=inst.data(mem->device_id);auto &s=inst.deferred(d);
        if(!s.enabled){d.cache.release(std::move(mem),!inst.no_cache_);return;}
        std::unique_lock<std::mutex> guard(s.lock);
        // Requested bytes describe logical live storage; reserved bytes continue
        // to include physically retained buffers until their fence completes.
        {
            std::unique_lock<std::mutex> cache_guard(d.cache.lock);
            d.cache.requested_size-=mem->orig_size;
        }
        s.bytes+=mem->size;s.peak_bytes=std::max(s.peak_bytes,s.bytes);++s.retired_count;
        s.staged.push_back(std::move(mem));
        if(s.bytes>=s.byte_limit)s.throttle(d.queue,d.cache);
    }

    void CLContextManager::sync_if_needed(int index)
    {
        auto &inst=instance();if(!inst.no_cache_)return;
        auto &d=inst.data(index);auto &s=inst.deferred(d);
        std::unique_lock<std::mutex> guard(s.lock);++s.sync_calls;
        if(!s.enabled) {
            auto start=std::chrono::steady_clock::now();d.queue.finish();++s.finish_calls;
            if(s.timing)s.host_wait_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
            return;
        }
        s.seal(d.queue);s.collect(d.cache,false);
        if(++s.ops>=s.op_limit || s.bytes>=s.byte_limit)s.throttle(d.queue,d.cache);
    }

    void CLContextManager::drain_deferred(int index,bool wait)
    {
        auto &inst=instance();auto &d=inst.data(index);
        if(!d.deferred || !d.deferred->enabled)return;
        auto &s=*d.deferred;std::unique_lock<std::mutex> guard(s.lock);
        s.seal(d.queue);s.collect(d.cache,wait);
        if(wait)s.ops=0;
    }

    std::map<std::string,int64_t> CLContextManager::memory_stats(int index,bool reset)
    {
        TORCH_CHECK(index>=0 && index<int(count()),"Invalid OpenCL device index: ",index);
        auto &inst=instance();auto &d=inst.data(index);auto &s=inst.deferred(d);
        std::unique_lock<std::mutex> guard(s.lock);
        if(s.enabled){s.seal(d.queue);s.collect(d.cache,false);}
        auto stats=d.cache.memory_stats(reset);
        stats["deferred_bytes.all.current"]=s.bytes;
        stats["deferred_bytes.all.peak"]=s.peak_bytes;
        stats["deferred.allocations"]=s.retired_count;
        stats["queue.sync_calls"]=s.sync_calls;
        stats["queue.finish_calls"]=s.finish_calls;
        stats["queue.host_wait_ns"]=s.host_wait_ns;
        stats["queue.marker_batches"]=s.markers;
        stats["queue.flushes"]=s.flushes;
        stats["queue.throttles"]=s.throttles;
        if(reset)s.peak_bytes=s.bytes;
        return stats;
    }

    bool CLContextManager::bad_fork_ = false;

    void CLContextManager::stop_profiling(int device,std::string const &output)
    {
        auto &data = instance().data(device);
        if(!data.enable_profiling || !data.timing) {
            throw std::runtime_error("You must enable profiling: torch.ocl.enable_profiling(device) and call stop after finishing ");
        }
        data.queue.finish();
        ExecGuard::set_profiling_context(nullptr);
        std::shared_ptr<dlprim::TimingData> timing = data.timing;
        data.queue.enable_timing(nullptr);
        if(output.empty()) {
            return;
        }
        std::ofstream log(output);
        log << "section,kernel,start (ms),end (ms),duraion(ms)\n";
        double point0 = -1.0;
        for(auto &d : timing->events()) {
            try {
                auto end_ms   = d->event.getProfilingInfo<CL_PROFILING_COMMAND_END>() * 1e-6;
                auto start_ms = d->event.getProfilingInfo<CL_PROFILING_COMMAND_START>() * 1e-6;
                if(point0 == -1)
                    point0 = start_ms;
                double time_ms = (end_ms - start_ms);
                int s = d->section;
                std::stack<char const *> sections;
                while(s!=-1) {
                    auto &sec = timing->sections().at(s);
                    sections.push(sec.name);
                    s=sec.parent;
                }
                while(!sections.empty()) {
                    log << sections.top();
                    sections.pop();
                    if(!sections.empty())
                        log<<":";
                }
                log<<"," << d->name;
                if(d->index != -1)
                     log << '[' << d->index << ']';
                log << "," << (start_ms-point0)<<","<<(end_ms-point0) << ","  << time_ms << "\n";
            }
            catch(cl::Error const &e) {
                log << "Failed for " << d->name << " " << e.what() << e.err() << std::endl;
            }
        }
    }
    void CLContextManager::start_profiling(int device)
    {
        auto &data = instance().data(device);
        if(!data.enable_profiling) {
            throw std::runtime_error("You must enable profiling: torch.ocl.enable_profiling(device)");
        }
        data.queue.finish();
        data.timing.reset(new dlprim::TimingData());
        data.queue.enable_timing(data.timing);
        ExecGuard::set_profiling_context(&data.queue);
    }
} // namespace

