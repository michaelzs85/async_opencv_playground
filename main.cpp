
#include <experimental/filesystem>
#include <fmt/format.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <future>
#include <cmath>
#include <unistd.h>

template <typename T>
int roundtoint(T val)
{
  return static_cast<int>(std::round(val));
}

using namespace std::experimental::filesystem;

struct is_image_impl
{
  std::unordered_set<std::string> const common_extensions{".jpg", ".png", ".jpeg"};

  bool operator()(path const& file)
  {
    return std::any_of(std::begin(common_extensions),
                       std::end(common_extensions),
                       [&file] (std::string_view known_extension)
                        {
                          return file.has_extension() && file.extension() == known_extension;
                        }
                      );
  }
};
static is_image_impl is_image;

struct config
{
  path outputdir;
  cv::Size blur_kernel;
};

cv::Mat blur(std::future<cv::Mat> f_in, config const& c)
{
  cv::Mat in = f_in.get();
  cv::Mat out;
  cv::blur(in, out, c.blur_kernel);
  return out;
}

cv::Mat half(std::future<cv::Mat> f_in, config const&)
{
  cv::Mat in = f_in.get();
  cv::Size newsize(roundtoint(0.5*in.cols), roundtoint(0.5*in.rows));
  cv::Mat out;
  cv::resize(in, out, newsize);
  return out;
}

cv::Mat save(std::future<cv::Mat> f_in, config const& c)
{
  static int i{1};
  cv::Mat in = f_in.get();
  cv::imwrite(c.outputdir.string() + fmt::format("/{}.jpg", i++), in);
  return {};
}

int main(int, char*[])
{
  path p("/home/michaelzs/Pictures/");
  auto directory_entries = directory_iterator(p);
  std::vector<path> images;
  std::for_each(begin(directory_entries), end(directory_entries), [&images](directory_entry const& di){
    if( is_image(di.path()) )
    {
      images.push_back(di.path());
    }
  });

  config c{"/tmp/resized", cv::Size(5,100)};
  std::vector<cv::Mat(*)(std::future<cv::Mat>, config const&)> worksteps{blur, half, blur, blur, half, save};

  std::vector<std::future<cv::Mat>> futures(images.size());
  for(path p : images)
  {
    std::future<cv::Mat> tmp = std::async(std::launch::async, [](path p){
        return cv::imread(p.string());
    }, p);
    for(auto func : worksteps)
    {
      tmp = std::async(std::launch::async, func, std::move(tmp), c);
    }
    futures.push_back(std::move(tmp));
  }

  return 0;
}
