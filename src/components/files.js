function modelCode(props){
    return props.imageSizeX+`,`+props.imageSizeY+`import numpy as np

    import torch
    import torch.nn as nn
    
    class PatchEmbed(nn.Module):
        
        def __init__(self, img_size=256, patch_size=32, window_size=4, in_chans=2, embed_dim=2048):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.n_patches = (img_size // patch_size) ** 2
            self.window_size = window_size
            self.embed_dim = embed_dim
            
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
            
        def forward(self, x):   # (batch_size, n_windows, in_chans, img_size, img_size) > (32, 4, 2, 256, 256)
            x = x.flatten(0,1)  # (batch_size * n_windows, in_chans, img_size, img_size) > (32*4, 2, 256, 256)
            x = self.proj(x)    # (batch_size * n_windows, embed_dim, n_patches ** 0.5, n_patches ** 0.5) > (32*4, 2048, 8, 8)
            x = x.flatten(2)    # (batch_size * n_windows, embed_dim, n_patches) > (32*4, 2048, 64)
            x = x.reshape(-1, self.window_size, self.embed_dim, self.n_patches) # > (32, 4, 2048, 64)
            x = x.transpose(2,3).flatten(1,2) # (batch_size, n_windows * n_patches, embed_dim) > (32, 4 * 64, 2048) 
    
            return x
        
        
    class Attention(nn.Module):
        
        def __init__(self, dim, n_heads=16, qkv_bias = True, attn_p=0., proj_p=0.):
            super().__init__()
            self.n_heads = n_heads
            self.dim = dim
            self.head_dim = dim // n_heads
            self.scale = self.head_dim ** -0.5
            
            self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
            self.attn_drop = nn.Dropout(attn_p)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_p)
            
        def forward(self, x):    # (n_samples, n_patches , dim) > (4, 64, 2048)
            n_samples, n_tokens, dim = x.shape
            
            if dim != self.dim:
                raise ValueError
            
            qkv = self.qkv(x)    # (n_samples, n_patches , 3 * dim) > (4, 64, 3*2048)
            qkv = qkv.reshape(   
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
            )                    # (n_samples, n_patches , 3 , n_heads, head_dim) > (4, 64, 3, 16, 128)
            qkv = qkv.permute(2,0,3,1,4) # (3, n_samples, n_heads, n_patches, head_dim) > (3, 4, 16, 64, 128)
            
            q, k, v = qkv[0], qkv[1], qkv[2]
            k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches) > (4, 16, 128, 64)
            
            dp = (
                q @ k_t
            ) * self.scale      # (n_samples, n_heads, n_patches, n_patches) > (4, 16, 64, 64)
            
            attn = dp.softmax(dim = -1)      # (n_samples, n_heads, n_patches, n_patches) > (4, 16, 64, 64)
            attn = self.attn_drop(attn)
            
            weighted_avg = attn @ v # (n_samples, n_heads, n_patches, head_dim) > (4, 16, 64, 128)
            weighted_avg = weighted_avg.transpose(1,2) # (n_samples, n_patches, n_heads, head_dim) > (4, 64, 16, 128)
            weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches, dim) > (4, 64, 2048)
            
            x = self.proj(weighted_avg) # (n_samples, n_patches, dim) > (4, 64, 2048)
            x = self.proj_drop(x) # (n_samples, n_patches, dim) > (4, 64, 2048)
            
            return x
    
        
    class MLP(nn.Module):
        
        def __init__(self, in_features, hidden_features, out_features, p=0.):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(p)
            
        def forward(self, x): # (n_samples, n_patches, in_features) > (4, 64, 2048)
            x = self.fc1(x)    # (n_samples, n_patches, hidden_features) > (4, 64, 512)
            x = self.drop(self.act(x))
            x = self.fc2(x)    # (n_samples, n_patches, out_features) > (4, 64, 2048)
            x = self.drop(x)
            return x
        
        
    class Block(nn.Module):
        
        def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias = True, p=0., attn_p = 0.):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
            )
            self.norm2= nn.LayerNorm(dim, eps=1e-6)
            hidden_features = int(dim * mlp_ratio)
            self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
            )
            
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))    
            
            return x
    
        
    class Decoder(nn.Module):
        
        def __init__(self, img_size=256, patch_size=32, window_size=4, in_chans=2, embed_dim=2048):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.n_patches = (img_size // patch_size) ** 2
            self.embed_dim = embed_dim
            self.window_size = window_size
            self.in_chans = in_chans
            
            self.proj = nn.ConvTranspose2d(
                embed_dim,
                in_chans,
                kernel_size=patch_size,
                stride=patch_size,
            )
            
        def forward(self, x):
            #print(x.shape) # (32, 4 * 64, 2048)
            x = x.reshape(-1, self.window_size, self.n_patches, self.embed_dim) # > (32, 4, 64, 2048)
            x = x.transpose(2,3) # > (32, 4, 2048, 64)
            x = x.flatten(0,1) # > (32*4, 2048, 64)
            x = x.view(-1, self.embed_dim, int(self.n_patches**0.5), int(self.n_patches**0.5)) # > (32*4, 2048, 8, 8)
            x = self.proj(x) # (n_samples, n_patches, embed_dim) > (32*4, 2, 256, 256)
            x = x.reshape(-1, self.window_size, self.in_chans, self.img_size, self.img_size) # > (32, 4, 2, 256, 256)
    
            return x
        
        
    class EventTransformer(nn.Module):
        def __init__(self,
                     img_size=256,
                     patch_size=32,
                     in_chans=2,
                     window_size=4,
                     embed_dim=2048,
                     depth=12,
                     n_heads=16,
                     mlp_ratio=4.0,
                     qkv_bias=True,
                     p=0.,
                     attn_p=0.,
                    ):
            super().__init__()
            
            self.patch_embed = PatchEmbed(
                img_size = img_size,
                patch_size = patch_size,
                in_chans = in_chans,
                window_size = window_size,
                embed_dim = embed_dim,
            )
            
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.n_patches * window_size, embed_dim)
            )
            self.pos_drop = nn.Dropout(p=p)
            
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim = embed_dim,
                        n_heads = n_heads,
                        mlp_ratio = mlp_ratio,
                        qkv_bias = qkv_bias,
                        p = p,
                        attn_p = attn_p,
                    )
                    for _ in range(depth)
                ]
            )
            
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
            self.decoder = Decoder(
                img_size = img_size,
                patch_size = patch_size,
                in_chans = in_chans,
                window_size = window_size,
                embed_dim = embed_dim,
            )
             
        def forward(self, x):
            n_samples = x.shape[0]
            x = self.patch_embed(x)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            
            for block in self.blocks:
                x = block(x)
                
            x = self.norm(x)
            
            x = self.decoder(x)
            return x
`};

function trainCode(props){
    return props.imageSizeX+`,`+props.imageSizeY+`import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
`+
((props.imageSizeX==="244") ?
`from torch.utils.tensorboard import SummaryWriter`
:
``)
+
`
import matplotlib.pyplot as plt
`};

const someHTMLCodeExample = `
  <!DOCTYPE html>
  <html lang="en">
    <head>
      <meta charset="utf-8" />
      <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
      <!-- https://web.dev/uses-rel-preconnect -->
      <link rel="preconnect" href="https://storage.googleapis.com">
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <meta name="theme-color" content="#111" />

      <meta
        name="description"
        content="Wlist"
        data-react-helmet="true"
      />
      <meta
        property="og:title"
        content="Wlist"
        data-react-helmet="true"
      >
      <meta
        property="og:description"
        content="Wlist"
        data-react-helmet="true"
      >
      <meta
        property="og:url"
        content="%PUBLIC_URL%"
        data-react-helmet="true"
      >
      <meta
        property="og:image"
        content="%PUBLIC_URL%/images/cover.png"
        data-react-helmet="true"
      />
      <meta
        name="twitter:card"
        content="summary"
        data-react-helmet="true"
      />
      <meta property="og:type" content="website" />
      <link rel="apple-touch-icon" href="%PUBLIC_URL%/logo192.png" />
      <!--
        manifest.json provides metadata used when your web app is installed on a
        user's mobile device or desktop. See https://developers.google.com/web/fundamentals/web-app-manifest/
      -->
      <link rel="manifest" href="%PUBLIC_URL%/manifest.json" crossorigin="use-credentials" />
      <!-- https://web.dev/defer-non-critical-css/ -->
      <link rel="preload" href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap" as="style" onload="this.onload=null;this.rel='stylesheet'">

      <title>Wlist</title>

      <!-- ie -->
      <script type="text/javascript">
        var ua = navigator.userAgent;
        var is_ie = ua.indexOf('MSIE ') > -1 || ua.indexOf('Trident/') > -1;

        if (is_ie) {
          document.ie = 'true';

          var ie_script = document.createElement('script');
          var ie_styles = document.createElement('link');

          ie_script.src = 'no-ie/init.js';
          ie_styles.rel = 'stylesheet';
          ie_styles.href = 'no-ie/styles.css';

          function injectScripts() {
            document.body.innerHTML = '';
            document.body.appendChild(ie_styles);
            document.body.appendChild(ie_script);
          }

          if (document.addEventListener) {
            document.addEventListener('DOMContentLoaded', injectScripts);
          } else { // before IE 9
            document.attachEvent('DOMContentLoaded', injectScripts);
          }

        }
      </script>
    </head>
    <body>
      <noscript>You need to enable JavaScript to run this app.</noscript>
      <script type="text/javascript">
        // set the body color before app initialization, to avoid blinking
        var themeMode = localStorage.getItem('theme-mode');
        var initialBodyStyles = document.createElement('style');
        var currentThemeColor = themeMode === 'light' ? '#fafafa': '#111';
        initialBodyStyles.innerText = 'body { background-color: ' + currentThemeColor + ' }';
        document.head.appendChild(initialBodyStyles);

        // also set meta[name="theme-color"] content
        var metaTheme = document.querySelector('meta[name="theme-color"]');

        metaTheme.content = currentThemeColor;
      </script>
      <div id="root"></div>
    </body>
  </html>
`;

function files(props) {

    const file = {
    "model.py": {
        name: "model.py",
        language: "python",
        value: modelCode(props)
    },
    "train.py": {
        name: "train.py",
        language: "python",
        value: trainCode(props)
    },
    "index.html": {
        name: "index.html",
        language: "html",
        value: someHTMLCodeExample
    }
    };

    return file[props.fileName];
}

export default files;
