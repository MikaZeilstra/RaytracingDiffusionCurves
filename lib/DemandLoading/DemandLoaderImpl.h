//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <DemandLoading/DemandLoader.h>

#include "Memory/DeviceMemoryManager.h"
#include "Memory/PinnedMemoryManager.h"
#include "PageTableManager.h"
#include "PagingSystem.h"
#include "RequestProcessor.h"
#include "ResourceRequestHandler.h"
#include "Textures/DemandTextureImpl.h"
#include "Textures/SamplerRequestHandler.h"

#include <cuda.h>

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace demandLoading {

struct DeviceContext;
class DemandTexture;
class ImageReader;
class RequestProcessor;
struct TextureDescriptor;

/// DemandLoader demonstrates how to implement demand-loaded textures using the OptiX paging library.
class DemandLoaderImpl : public DemandLoader
{
  public:
    /// Construct demand loading sytem.
    DemandLoaderImpl( const std::vector<unsigned int>& devices, const Options& options );

    /// Destroy demand loading system.
    ~DemandLoaderImpl();

    /// Create a demand-loaded texture for the given image.  The texture initially has no backing
    /// storage.  The readTile() method is invoked on the image to fill each required tile.  The
    /// ImageReader pointer is retained indefinitely.
    const DemandTexture& createTexture( std::shared_ptr<ImageReader> image, const TextureDescriptor& textureDesc ) override;

    /// Create an arbitrary resource with the specified number of pages.  \see ResourceCallback.
    unsigned int createResource( unsigned int numPages, ResourceCallback callback ) override;

    /// Prepare for launch.  Returns false if the specified device does not support sparse textures.
    /// If successful, returns a DeviceContext via result parameter, which should be copied to
    /// device memory (typically along with OptiX kernel launch parameters), so that it can be
    /// passed to Tex2D().
    bool launchPrepare( unsigned int deviceIndex, CUstream stream, DeviceContext& demandTextureContext ) override;

    /// Fetch page requests from the given device context and enqueue them for background
    /// processing.  The given stream is used when copying tile data to the device.  Returns a
    /// ticket that is notified when the requests have been filled.
    std::shared_ptr<Ticket> processRequests( unsigned int deviceIndex, CUstream stream, const DeviceContext& deviceContext ) override;

    /// Get current statistics.
    Statistics getStatistics() const override;

    /// Get the demand loading configuration options.
    const Options& getOptions() const { return m_options; }

    /// Get the number of devices (including inactive devices, if any).
    unsigned int getNumDevices() const { return static_cast<unsigned int>( m_pagingSystems.size() ); }

    /// Check whether the specified device is active.
    bool isActiveDevice( unsigned int deviceIndex ) const
    {
        return static_cast<bool>( m_pagingSystems.at( deviceIndex ) );
    }

    /// Get the DeviceMemoryManager for the specified device.
    DeviceMemoryManager* getDeviceMemoryManager( unsigned int deviceIndex )
    {
        return m_deviceMemoryManagers[deviceIndex].get();
    }

    /// Get the pinned memory manager.
    PinnedMemoryManager* getPinnedMemoryManager() { return &m_pinnedMemoryManager; }

    /// Get the specified texture.
    DemandTextureImpl* getTexture( unsigned int textureId ) { return m_textures.at( textureId ).get(); }

    /// Get the PagingSystem for the specified device.
    PagingSystem* getPagingSystem( unsigned int deviceIndex ) { return m_pagingSystems.at( deviceIndex ).get(); }

    /// Get the PageTableManager.
    PageTableManager* getPageTableManager() { return &m_pageTableManager; }

    /// Unmap the backing storage associated with a texture tile or mip tail
    void unmapTileResource( unsigned int deviceIndex, CUstream stream, unsigned int pageId );

    // Free stale tiles as needed to keep free tiles in the tile pools
    void freeTiles( unsigned int deviceIndex, CUstream stream );

  private:
    mutable std::mutex m_mutex;
    Options            m_options;
    unsigned int       m_numDevices;

    std::vector<std::unique_ptr<DemandTextureImpl>>   m_textures;  // demand-loaded textures, indexed by texture id.
    std::vector<std::unique_ptr<DeviceMemoryManager>> m_deviceMemoryManagers;  // Manages device memory (one per device)
    std::vector<std::unique_ptr<PagingSystem>>        m_pagingSystems;  // Manages device interaction (one per device)

    SamplerRequestHandler m_samplerRequestHandler;  // Handles requests for texture samplers.
    PageTableManager      m_pageTableManager;       // Allocates ranges of virtual pages.
    RequestProcessor      m_requestProcessor;       // Asynchronously processes page requests.
    PinnedMemoryManager   m_pinnedMemoryManager;

    std::vector<std::unique_ptr<ResourceRequestHandler>> m_resourceRequestHandlers;  // Request handlers for arbitrary resources.

    double m_totalProcessingTime = 0.0;
};

}  // namespace demandLoading
