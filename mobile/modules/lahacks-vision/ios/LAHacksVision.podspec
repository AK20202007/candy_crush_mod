require "json"

package = JSON.parse(File.read(File.join(__dir__, "..", "package.json")))

Pod::Spec.new do |s|
  s.name         = "LAHacksVision"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.description  = package["description"]
  s.homepage     = "https://example.com/lahacks-vision"
  s.license      = package["license"]
  s.author       = "LAHacks"
  s.platforms    = { :ios => "15.1" }
  s.source       = { :git => "https://example.com/lahacks-vision.git", :tag => s.version.to_s }
  s.static_framework = true

  s.dependency "ExpoModulesCore"
  # Pod root is this `ios/` directory (see Podfile.lock `:path => ../modules/lahacks-vision/ios`).
  s.source_files = "**/*.{h,m,mm,swift}"
end
