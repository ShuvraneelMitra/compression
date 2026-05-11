#include "kl_transform.hpp"

cv::Mat1d find_block_mean(const cv::Mat& A, uint block_size) {
    /*
    IMPORTANT: Right now we are dealing with only square, grayscale images
    and the block_size MUST be a factor of the image size for the function 
    to work.
    */
    if(A.rows != A.cols) throw std::invalid_argument("Input image should be square and grayscale!");
    if(A.rows % block_size != 0) throw std::invalid_argument("Block size should be a factor of the image size");
    
    int n = A.rows;
    std::vector<double> mean_vec(block_size * block_size); //default value 0
    int d = n / block_size;
    int num_blocks = (n / block_size) * (n / block_size);

    for(int i = 0; i < d; i++){
        for(int j = 0; j < d; j++){
            /*
            The (i, j)-th block of size block_size x block_size starts from 
            row block_size * i and from column block_size * j. Considering these
            as the origin (0th row and col respectively), A[u][v] maps to 
            mean[u * block_size + v]
            */
            
            // SINGLE BLOCK PROCESSING
            for(int u = 0; u < block_size; u++){
                for(int v = 0; v < block_size; v++){
                    int r = block_size * i + u;
                    int c = block_size * j + v;

                    mean_vec[u * block_size + v] += A.at<double>(r, c);
                }
            }
        }
    }

    cv::Mat1d mean_mat = cv::Mat(mean_vec).clone();
    mean_mat /= num_blocks;
    mean_mat = mean_mat.reshape(1, 1);
    return mean_mat; 
}

cv::Mat1d find_img_cov(const cv::Mat& A, uint block_size) {

    cv::Mat1d mu = find_block_mean(A, block_size);

    int n = A.rows;
    int d = n / block_size;
    cv::Mat1d cov = cv::Mat1d::zeros(block_size * block_size, block_size * block_size);
    int num_blocks = (n / block_size) * (n / block_size);

    cv::Mat1d flattened(1, block_size * block_size);
    cv::Mat1d centered(1, block_size * block_size);
    cv::Mat1d outer_prod(block_size * block_size, block_size * block_size);

    for(int i = 0; i < d; i++){
        for(int j = 0; j < d; j++){

            // a different type of block processing
            cv::Rect block(j * block_size, i * block_size,
                            block_size, block_size);
            cv::Mat submatrix = A(block);

            flattened = submatrix.clone().reshape(1, 1); 
            centered = flattened - mu;
            outer_prod = centered.t() * centered;
            
            cov += outer_prod;
        }
    }

    cov /= (num_blocks - 1); // for unbiased estimator
    return cov;
}

cv::Mat1d ordered_diagonalization(const cv::Mat& A) {
    cv::Mat eigenvalues;
    cv::Mat eigenvectors;

    /*
    By a miracle we get exactly what we are looking for: a matrix with
    its rows containing the normalized eigenvectors arranged according
    to the descending order of their corresponding eigenvalues
    */
    cv::eigen(A, eigenvalues, eigenvectors);
    return eigenvectors;
}

std::tuple<cv::Mat1d, cv::Mat1d, cv::Mat1d, cv::Mat1d>
kl_transform(cv::Mat& img, uint block_size, uint top_k) {
    /*
    Returns 
    1. a matrix of size NUM_BLOCKS x top_k where each row defines
    the set of transform coefficients for each block of size block_size x block_size
    2. The covariance matrix of the original image
    3. The mu vector
    4. The gamma_topk matrix
    */

    if (top_k > static_cast<uint>(block_size * block_size))
        throw std::invalid_argument("top_k cannot exceed block_size^2");

    cv::Mat1d A;
    img.convertTo(A, CV_64F);

    int n = A.rows;
    int d = n / block_size;
    int num_blocks = (n / block_size) * (n / block_size);

    cv::Mat1d mu = find_block_mean(A, block_size);
    cv::Mat1d cov = find_img_cov(A, block_size);
    cv::Mat1d gamma = ordered_diagonalization(cov);

    cv::Mat1d gamma_topk = gamma.rowRange(0, top_k).clone();

    cv::Mat1d tf_coeffs = cv::Mat1d::zeros(num_blocks, top_k);
    int idx = 0;

    for(int i = 0; i < d; i++){
        for(int j = 0; j < d; j++){
            cv::Rect block(j * block_size, i * block_size,
                            block_size, block_size);
            cv::Mat1d submatrix = A(block);
            cv::Mat1d flattened = submatrix.clone().reshape(1, 1); 
            cv::Mat1d b = flattened - mu;
            cv::Mat1d y = gamma_topk * b.t();

            for(int k = 0; k < top_k; k++) {
                tf_coeffs(idx, k) = y.at<double>(k, 0);
            }
            idx++;
        }
    }
    return std::make_tuple(tf_coeffs, cov, mu, gamma_topk);
}

cv::Mat1b kl_reconstruct(uint img_dim, cv::Mat& coeffs, cv::Mat& mu, cv::Mat& gamma_topk, uint block_size) {
    // int n = static_cast<int>(std::sqrt(coeffs.rows)) * block_size;
    int n = img_dim;
    int d = n / block_size;
    int num_blocks = (n / block_size) * (n / block_size);

    cv::Mat1b recon = cv::Mat1b::zeros(n, n);
    cv::Mat1d reconstructed_block_matrix = gamma_topk.t() * coeffs.t();

    cv::Mat1d mu_col = mu.reshape(1, mu.total());
    cv::Mat repeated_mu = cv::repeat(mu_col, 1, reconstructed_block_matrix.cols);
    reconstructed_block_matrix += repeated_mu;

    for(int i = 0; i < d; i++){
        for(int j = 0; j < d; j++){
            int block_number = i * d + j;
            for(int u = 0; u < block_size; ++u){
                for(int v = 0; v < block_size; ++v){
                    int r = i * block_size + u;
                    int c = j * block_size + v;
                    
                    recon.at<uchar>(r, c) = static_cast<uchar>(
                                                    std::clamp(
                                                        std::round(
                                                            reconstructed_block_matrix(u * block_size + v, block_number)
                                                        ), 0.0, 255.0
                                                    )
                                            );
                }
            }
        }
    }

    return recon;
}